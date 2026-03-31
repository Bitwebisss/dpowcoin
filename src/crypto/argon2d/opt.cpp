/*
 * Argon2 reference source code package - reference C implementations
 *
 * Copyright 2015
 * Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, and Samuel Neves
 *
 * You may use this work under the terms of a Creative Commons CC0 1.0
 * License/Waiver or the Apache Public License 2.0, at your option. The terms of
 * these licenses can be found at:
 *
 * - CC0 1.0 Universal : https://creativecommons.org/publicdomain/zero/1.0
 * - Apache 2.0        : https://www.apache.org/licenses/LICENSE-2.0
 *
 * You should have received a copy of both of these licenses along with this
 * software. If not, they may be obtained at the above URLs.
 */

/*
 * x86 optimized path + runtime ISA selector for Argon2 fill_segment.
 * Non-x86 systems use ref.c exclusively (fill_segment / Argon2AutoDetectImpl).
 *
 * Structure mirrors sha256.cpp:
 *   - DISABLE_OPTIMIZED_ARGON2 is the outer kill switch; it gates cpuid.h
 *     and all optimized code, same as DISABLE_OPTIMIZED_SHA256 in sha256.cpp.
 *   - HAVE_GETCPUID (defined by cpuid.h on x86) is the inner x86 guard.
 *   - ENABLE_ARGON2_AVX2 / ENABLE_ARGON2_AVX512 gate separately-compiled
 *     ISA variants, same as ENABLE_AVX2 / ENABLE_SSE41 in sha256.cpp.
 */

#if !defined(DISABLE_OPTIMIZED_ARGON2)
#include <compat/cpuid.h>

#if defined(HAVE_GETCPUID)

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "argon2.h"
#include "core.h"
#include "blake2/blake2.h"
#include "blake2/blamka-round-opt.h"   /* x86 path: SSE2/__m128i fallback */

/* -------------------------------------------------------------------------
 * Forward declarations for separately-compiled ISA variants.
 * Mirrors the namespace forward-decls at the top of sha256.cpp.
 * ------------------------------------------------------------------------- */
#if defined(ENABLE_ARGON2_AVX512)
void fill_segment_avx512(const argon2_instance_t *instance,
                         argon2_position_t position);
#endif

#if defined(ENABLE_ARGON2_AVX2)
void fill_segment_avx2(const argon2_instance_t *instance,
                       argon2_position_t position);
#endif

/* -------------------------------------------------------------------------
 * SSE2 fill_block / next_addresses / fill_segment_sse2
 * This is the x86 baseline (SSE2 / __m128i), compiled without extra flags.
 * Equivalent to sha256_sse4::Transform — always present on x86, used when
 * AVX2 / AVX-512 are not available.
 * ------------------------------------------------------------------------- */
static void fill_block(__m128i *state, const block *ref_block,
                       block *next_block, int with_xor) {
    __m128i block_XY[ARGON2_OWORDS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            state[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)ref_block->v + i));
            block_XY[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)next_block->v + i));
        }
    } else {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            block_XY[i] = state[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)ref_block->v + i));
        }
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * i + 0], state[8 * i + 1], state[8 * i + 2],
            state[8 * i + 3], state[8 * i + 4], state[8 * i + 5],
            state[8 * i + 6], state[8 * i + 7]);
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i],
            state[8 * 3 + i], state[8 * 4 + i], state[8 * 5 + i],
            state[8 * 6 + i], state[8 * 7 + i]);
    }

    for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
        state[i] = _mm_xor_si128(state[i], block_XY[i]);
        _mm_storeu_si128((__m128i *)next_block->v + i, state[i]);
    }
}

static void next_addresses(block *address_block, block *input_block) {
    __m128i zero_block[ARGON2_OWORDS_IN_BLOCK];
    __m128i zero2_block[ARGON2_OWORDS_IN_BLOCK];
    memset(zero_block,  0, sizeof(zero_block));
    memset(zero2_block, 0, sizeof(zero2_block));
    input_block->v[6]++;
    fill_block(zero_block,  input_block,   address_block, 0);
    fill_block(zero2_block, address_block, address_block, 0);
}

static void fill_segment_sse2(const argon2_instance_t *instance,
                              argon2_position_t position) {
    block *ref_block = NULL, *curr_block = NULL;
    block address_block, input_block;
    uint64_t pseudo_rand, ref_index, ref_lane;
    uint32_t prev_offset, curr_offset;
    uint32_t starting_index, i;
    __m128i state[ARGON2_OWORDS_IN_BLOCK];
    int data_independent_addressing;

    if (instance == NULL) {
        return;
    }

    data_independent_addressing =
        (instance->type == Argon2_i) ||
        (instance->type == Argon2_id && (position.pass == 0) &&
         (position.slice < ARGON2_SYNC_POINTS / 2));

    if (data_independent_addressing) {
        init_block_value(&input_block, 0);
        input_block.v[0] = position.pass;
        input_block.v[1] = position.lane;
        input_block.v[2] = position.slice;
        input_block.v[3] = instance->memory_blocks;
        input_block.v[4] = instance->passes;
        input_block.v[5] = instance->type;
    }

    starting_index = 0;

    if ((0 == position.pass) && (0 == position.slice)) {
        starting_index = 2;
        if (data_independent_addressing) {
            next_addresses(&address_block, &input_block);
        }
    }

    curr_offset = position.lane * instance->lane_length +
                  position.slice * instance->segment_length + starting_index;

    if (0 == curr_offset % instance->lane_length) {
        prev_offset = curr_offset + instance->lane_length - 1;
    } else {
        prev_offset = curr_offset - 1;
    }

    memcpy(state, ((instance->memory + prev_offset)->v), ARGON2_BLOCK_SIZE);

    for (i = starting_index; i < instance->segment_length;
         ++i, ++curr_offset, ++prev_offset) {
        if (curr_offset % instance->lane_length == 1) {
            prev_offset = curr_offset - 1;
        }

        if (data_independent_addressing) {
            if (i % ARGON2_ADDRESSES_IN_BLOCK == 0) {
                next_addresses(&address_block, &input_block);
            }
            pseudo_rand = address_block.v[i % ARGON2_ADDRESSES_IN_BLOCK];
        } else {
            pseudo_rand = instance->memory[prev_offset].v[0];
        }

        ref_lane = ((pseudo_rand >> 32)) % instance->lanes;
        if ((position.pass == 0) && (position.slice == 0)) {
            ref_lane = position.lane;
        }

        position.index = i;
        ref_index = index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF,
                                ref_lane == position.lane);

        ref_block  = instance->memory + instance->lane_length * ref_lane + ref_index;
        curr_block = instance->memory + curr_offset;

        if (ARGON2_VERSION_10 == instance->version) {
            fill_block(state, ref_block, curr_block, 0);
        } else {
            fill_block(state, ref_block, curr_block, position.pass != 0);
        }
    }
}

/* -------------------------------------------------------------------------
 * AVXEnabled() — same helper as sha256.cpp
 * ------------------------------------------------------------------------- */
static int AVXEnabled(void)
{
    uint32_t a, d;
    __asm__("xgetbv" : "=a"(a), "=d"(d) : "c"(0));
    return (a & 6) == 6;
}

static int AVX512Enabled(void)
{
    uint32_t a, d;
    __asm__("xgetbv" : "=a"(a), "=d"(d) : "c"(0));
    return (a & 0xE6) == 0xE6;
}

/* -------------------------------------------------------------------------
 * argon2_fill_segment — global function pointer, default = SSE2 baseline.
 * On x86, Argon2AutoDetectImpl() upgrades this to AVX2 or AVX-512 if available.
 * fill_segment() is the public entry called by core.c — routes through ptr.
 * ------------------------------------------------------------------------- */
void (*argon2_fill_segment)(const argon2_instance_t *instance,
                            argon2_position_t position) = fill_segment_sse2;

void fill_segment(const argon2_instance_t *instance,
                  argon2_position_t position) {
    argon2_fill_segment(instance, position);
}

/* -------------------------------------------------------------------------
 * Argon2AutoDetectImpl — x86 implementation.
 * Non-x86 stub is in ref.c.
 * ------------------------------------------------------------------------- */
const char *Argon2AutoDetectImpl(void)
{
    const char *ret = "sse2";
    argon2_fill_segment = fill_segment_sse2;

    {
        uint32_t eax, ebx, ecx, edx;
        int have_xsave, have_avx, enabled_avx;
        int have_avx2, have_avx512f;

        GetCPUID(1, 0, eax, ebx, ecx, edx);
        have_xsave = (ecx >> 27) & 1;
        have_avx   = (ecx >> 28) & 1;
        enabled_avx = 0;
        if (have_xsave && have_avx) {
            enabled_avx = AVXEnabled();
        }

        GetCPUID(7, 0, eax, ebx, ecx, edx);
        have_avx2    = (ebx >> 5)  & 1;
        have_avx512f = (ebx >> 16) & 1;

#if defined(ENABLE_ARGON2_AVX512)
        if (have_avx512f && have_xsave && AVX512Enabled()) {
            argon2_fill_segment = fill_segment_avx512;
            ret = "avx512";
        } else
#endif
#if defined(ENABLE_ARGON2_AVX2)
        if (have_avx2 && have_avx && enabled_avx) {
            argon2_fill_segment = fill_segment_avx2;
            ret = "avx2";
        } else
#endif
        {
            /* SSE2 baseline already set above */
            (void)have_avx; (void)have_avx2; (void)have_avx512f; (void)enabled_avx;
        }
    }

    return ret;
}

#endif /* HAVE_GETCPUID */
#endif /* !DISABLE_OPTIMIZED_ARGON2 */