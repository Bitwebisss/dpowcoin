// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <pow.h>

#include <arith_uint256.h>
#include <chain.h>
#include <primitives/block.h>
#include <uint256.h>

unsigned int GetNextWorkRequired(const CBlockIndex* pindexLast, const CBlockHeader *pblock, const Consensus::Params& params)
{
	
    assert(pindexLast != nullptr);
	/*
    
    // Only change once per difficulty adjustment interval
    if ((pindexLast->nHeight+1) % params.DifficultyAdjustmentInterval() != 0)
    {
        if (params.fPowAllowMinDifficultyBlocks)
        {
            // Special difficulty rule for testnet:
            // If the new block's timestamp is more than 2* 10 minutes
            // then allow mining of a min-difficulty block.
            if (pblock->GetBlockTime() > pindexLast->GetBlockTime() + params.nPowTargetSpacing*2)
                return nProofOfWorkLimit;
            else
            {
                // Return the last non-special-min-difficulty-rules-block
                const CBlockIndex* pindex = pindexLast;
                while (pindex->pprev && pindex->nHeight % params.DifficultyAdjustmentInterval() != 0 && pindex->nBits == nProofOfWorkLimit)
                    pindex = pindex->pprev;
                return pindex->nBits;
            }
        }
        return pindexLast->nBits;
    }
    
    // Go back by what we want to be 14 days worth of blocks
	

	
    int nHeightFirst = pindexLast->nHeight - (params.DifficultyAdjustmentInterval()-1);
    assert(nHeightFirst >= 0);

    if (nHeightFirst < 0) {
        return nProofOfWorkLimit;
    }

    const CBlockIndex* pindexFirst = pindexLast->GetAncestor(nHeightFirst);
    assert(pindexFirst);
    */
    return Lwma3CalculateNextWorkRequired(pindexLast, params);
}

unsigned int CalculateNextWorkRequired(const CBlockIndex* pindexLast, int64_t nFirstBlockTime, const Consensus::Params& params)
{
    if (params.fPowNoRetargeting)
        return pindexLast->nBits;

    // Limit adjustment step
    int64_t nActualTimespan = pindexLast->GetBlockTime() - nFirstBlockTime;
    if (nActualTimespan < params.nPowTargetTimespan/4)
        nActualTimespan = params.nPowTargetTimespan/4;
    if (nActualTimespan > params.nPowTargetTimespan*4)
        nActualTimespan = params.nPowTargetTimespan*4;

    // Retarget
    const arith_uint256 bnPowLimit = UintToArith256(params.powLimit);
    arith_uint256 bnNew;
    bnNew.SetCompact(pindexLast->nBits);
    bnNew *= nActualTimespan;
    bnNew /= params.nPowTargetTimespan;

    if (bnNew > bnPowLimit)
        bnNew = bnPowLimit;

    return bnNew.GetCompact();
}

// LWMA-1 for BTC & Zcash clones
// Copyright (c) 2017-2019 The Bitcoin Gold developers, Zawy, iamstenman (Microbitcoin)
// MIT License
// Algorithm by Zawy, a modification of WT-144 by Tom Harding
// For updates see
// https://github.com/zawy12/difficulty-algorithms/issues/3#issuecomment-442129791
// Do not use Zcash's / Digishield's method of ignoring the ~6 most recent 
// timestamps via the median past timestamp (MTP of 11).
// Changing MTP to 1 instead of 11 enforces sequential timestamps. Not doing this was the
// most serious, problematic, & fundamental consensus theory mistake made in bitcoin but
// this change may require changes elsewhere such as creating block headers or what pools do.
//  FTL should be lowered to about N*T/20.
//  FTL in BTC clones is MAX_FUTURE_BLOCK_TIME in chain.h.
//  FTL in Ignition, Numus, and others can be found in main.h as DRIFT.
//  FTL in Zcash & Dash clones need to change the 2*60*60 here:
//  if (block.GetBlockTime() > nAdjustedTime + 2 * 60 * 60)
//  which is around line 3700 in main.cpp in ZEC and validation.cpp in Dash
//  If your coin uses median network time instead of node's time, the "revert to 
//  node time" rule (70 minutes in BCH, ZEC, & BTC) should be reduced to FTL/2 
//  to prevent 33% Sybil attack that can manipulate difficulty via timestamps. See:
// https://github.com/zcash/zcash/issues/4021

unsigned int Lwma3CalculateNextWorkRequired(const CBlockIndex* pindexLast, const Consensus::Params& params)
{
    const int64_t T = params.nPowTargetSpacing;

  // For T=600 use N=288 (takes 2 days to fully respond to hashrate changes) and has 
  //  a StdDev of N^(-0.5) which will often be the change in difficulty in N/4 blocks when hashrate is 
  // constant. 10% of blocks will have an error >2x the StdDev above or below where D should be. 
  //  This N=288 is like N=144 in ASERT which is N=144*ln(2)=100 in 
  // terms of BCH's ASERT.  BCH's ASERT uses N=288 which is like 2*288/ln(2) = 831 = N for 
  // LWMA. ASERT and LWMA are almost indistinguishable once this adjustment to N is used. In other words,
  // 831/144 = 5.8 means my N=144 recommendation for T=600 is 5.8 times faster but SQRT(5.8) less 
  // stability than BCH's ASERT. The StdDev for 288 is 6%, so 12% accidental variation will be see in 10% of blocks.
  // Twice 288 is 576 which will have 4.2% StdDev and be 2x slower. This is reasonable for T=300 or less.
  // For T = 60, N=1,000 will have 3% StdDev & maybe plenty fast, but require 1M multiplications & additions per 
  // 1,000 blocks for validation which might be a consideration. I would not go over N=576 and prefer 360
  // so that it can respond in 6 hours to hashrate changes.

    const int64_t N = params.lwmaAveragingWindow;
	
	// Low diff blocks for diff initiation.
	const int64_t L = 577;

    // Define a k that will be used to get a proper average after weighting the solvetimes.
    const int64_t k = N * (N + 1) * T / 2; 

    const int64_t height = pindexLast->nHeight;
    const arith_uint256 powLimit = UintToArith256(params.powLimit);
    
   // New coins just "give away" first N blocks. It's better to guess
   // this value instead of using powLimit, but err on high side to not get stuck.
    if (params.fPowAllowMinDifficultyBlocks) { return powLimit.GetCompact(); }
    if (height <= L) { return powLimit.GetCompact(); }

    arith_uint256 avgTarget, nextTarget;
    int64_t thisTimestamp, previousTimestamp;
    int64_t sumWeightedSolvetimes = 0, j = 0;

    const CBlockIndex* blockPreviousTimestamp = pindexLast->GetAncestor(height - N);
    previousTimestamp = blockPreviousTimestamp->GetBlockTime();

    // Loop through N most recent blocks. 
    for (int64_t i = height - N + 1; i <= height; i++) {
        const CBlockIndex* block = pindexLast->GetAncestor(i);

        // Prevent solvetimes from being negative in a safe way. It must be done like this. 
        // Do not attempt anything like  if (solvetime < 1) {solvetime=1;}
        // The +1 ensures new coins do not calculate nextTarget = 0.
        thisTimestamp = (block->GetBlockTime() > previousTimestamp) ? 
                            block->GetBlockTime() : previousTimestamp + 1;

       // 6*T limit prevents large drops in diff from long solvetimes which would cause oscillations.
        int64_t solvetime = std::min(6 * T, thisTimestamp - previousTimestamp);

       // The following is part of "preventing negative solvetimes". 
        previousTimestamp = thisTimestamp;
       
       // Give linearly higher weight to more recent solvetimes.
        j++;
        sumWeightedSolvetimes += solvetime * j; 

        arith_uint256 target;
        target.SetCompact(block->nBits);
        avgTarget += target / N / k; // Dividing by k here prevents an overflow below.
    }
    nextTarget = avgTarget * sumWeightedSolvetimes; 

    if (nextTarget > powLimit) { nextTarget = powLimit; }

    return nextTarget.GetCompact();
}


// Check that on difficulty adjustments, the new difficulty does not increase
// or decrease beyond the permitted limits.
/*
bool PermittedDifficultyTransition(const Consensus::Params& params, int64_t height, uint32_t old_nbits, uint32_t new_nbits)
{
    
    if (params.fPowAllowMinDifficultyBlocks) return true;

    if (height % params.DifficultyAdjustmentInterval() == 0) {
        int64_t smallest_timespan = params.nPowTargetTimespan/4;
        int64_t largest_timespan = params.nPowTargetTimespan*4;

        const arith_uint256 pow_limit = UintToArith256(params.powLimit);
        arith_uint256 observed_new_target;
        observed_new_target.SetCompact(new_nbits);

        // Calculate the largest difficulty value possible:
        arith_uint256 largest_difficulty_target;
        largest_difficulty_target.SetCompact(old_nbits);
        largest_difficulty_target *= largest_timespan;
        largest_difficulty_target /= params.nPowTargetTimespan;

        if (largest_difficulty_target > pow_limit) {
            largest_difficulty_target = pow_limit;
        }

        // Round and then compare this new calculated value to what is
        // observed.
        arith_uint256 maximum_new_target;
        maximum_new_target.SetCompact(largest_difficulty_target.GetCompact());
        if (maximum_new_target < observed_new_target) return false;

        // Calculate the smallest difficulty value possible:
        arith_uint256 smallest_difficulty_target;
        smallest_difficulty_target.SetCompact(old_nbits);
        smallest_difficulty_target *= smallest_timespan;
        smallest_difficulty_target /= params.nPowTargetTimespan;

        if (smallest_difficulty_target > pow_limit) {
            smallest_difficulty_target = pow_limit;
        }

        // Round and then compare this new calculated value to what is
        // observed.
        arith_uint256 minimum_new_target;
        minimum_new_target.SetCompact(smallest_difficulty_target.GetCompact());
        if (minimum_new_target > observed_new_target) return false;
    } else if (old_nbits != new_nbits) {
        return false;
    }
    
    return true;
}
*/
// ---------------------------------------------------------------------------
// PermittedDifficultyTransition — restored for LWMA3 + Dual PoW
// ---------------------------------------------------------------------------
//
// PURPOSE:
//   Called by HeadersSyncState (headerssync.cpp) during PRESYNC and REDOWNLOAD
//   phases for EVERY single block. Provides anti-DoS protection: rejects chains
//   where nBits changes in ways that are mathematically impossible under LWMA3,
//   preventing an attacker from feeding fake header chains.
//
// WHY THE ORIGINAL BITCOIN VERSION DOES NOT WORK HERE:
//   Bitcoin checks difficulty only at 2016-block intervals (±4x per interval),
//   and requires identical nBits between intervals. With LWMA3, difficulty
//   adjusts every block. The original `else if (old_nbits != new_nbits)` branch
//   would reject every block after the first, completely breaking sync.
//
// LWMA3 MATHEMATICAL UPPER BOUND (the critical anti-DoS check):
//
//   nextTarget = avgTarget * sumWeightedSolvetimes
//   avgTarget  = Σ(target_i / N / k),   k = N*(N+1)*T/2
//
//   The key line in the algorithm:
//     solvetime = std::min(6 * T, thisTimestamp - previousTimestamp);
//
//   Maximum possible sumWeightedSolvetimes (all N solvetimes at their cap 6*T):
//     sumWeightedSolvetimes_max = 6*T * N*(N+1)/2 = 6*k
//     nextTarget_max = avgTarget * 6
//
//   Therefore: new_target can NEVER exceed old_target * 6.
//   This is a hard mathematical guarantee from the algorithm itself.
//   No legitimate block in this network can violate this. An attacker
//   presenting a header with new_target > old_target * 6 is lying about
//   their nBits, and we reject them.
//
//   Real per-block maximum for adjacent blocks (N=288, T=300):
//     ratio_max = 1 + 2*(6T-1)/((N+1)*T) ≈ 1.0415 (4.15%)
//   We use 6x (the whole-window theoretical maximum) — conservative and safe.
//
// WHY WE DO NOT CHECK A LOWER BOUND:
//
//   A lower bound would prevent target from dropping too fast (difficulty rising
//   too fast). However, this attack vector is already fully covered by
//   CheckProofOfWork: if an attacker claims very high difficulty (very low
//   target), they must actually produce a valid hash below that target —
//   which requires real hashpower they don't have. Adding a lower bound here
//   provides no additional security and risks false positives on historical
//   blocks from this existing network.
//
// BOOTSTRAPPING WINDOW (L=577, must match Lwma3CalculateNextWorkRequired):
//
//   LWMA3 logic: if (pindexLast->nHeight <= 577) return powLimit
//   For block at height h: pindexLast->nHeight = h-1
//     h-1 <= 577  =>  h <= 578  =>  blocks at heights 1..578 have nBits = powLimit
//
//   In PermittedDifficultyTransition, 'height' = height of the NEW block:
//     height <= 578: new_nbits MUST equal powLimit (bootstrapping window)
//     height >= 579: LWMA3 is active, apply 6x upper bound check
//
//   The transition at height 579 (first LWMA3 block) is handled automatically:
//     old_target = powLimit, max_permitted = min(powLimit*6, powLimit) = powLimit
//     LWMA3 always caps: if (nextTarget > powLimit) nextTarget = powLimit
//     So new_target <= powLimit = max_permitted — always satisfied.
//     No special-case branch needed for height 579.
//
// DUAL PoW (Yespower + Argon2id):
//
//   Both algorithms share the SAME nBits target. A block is valid only when:
//     hash_yespower  <= target  AND  hash_argon2id <= target
//   P(valid per nonce) = (target/2^256)^2
//
//   LWMA3 observes real block times, which already reflect the dual-PoW
//   probability. The 6x upper bound derived from the solvetime cap is
//   independent of the number of PoW algorithms — it holds for both.
//   One nBits check here covers both algorithms simultaneously.
//
// ---------------------------------------------------------------------------
 
bool PermittedDifficultyTransition(const Consensus::Params& params, int64_t height, uint32_t old_nbits, uint32_t new_nbits)
{
    // Testnet/regtest: min-difficulty blocks are permitted, any transition is valid
    if (params.fPowAllowMinDifficultyBlocks) return true;
 
    bool fNegative, fOverflow;
    const arith_uint256 pow_limit = UintToArith256(params.powLimit);
    const uint32_t pow_limit_compact = pow_limit.GetCompact();
 
    // Validate old_nbits: must be a well-formed, in-range target
    arith_uint256 old_target;
    old_target.SetCompact(old_nbits, &fNegative, &fOverflow);
    if (fNegative || fOverflow || old_target == 0 || old_target > pow_limit) {
        return false;
    }
 
    // Validate new_nbits: must be well-formed
    arith_uint256 new_target;
    new_target.SetCompact(new_nbits, &fNegative, &fOverflow);
    if (fNegative || fOverflow || new_target == 0) {
        return false;
    }
 
    // Hard ceiling: target can never exceed powLimit under any circumstances
    if (new_target > pow_limit) {
        return false;
    }
 
    // -------------------------------------------------------------------
    // LWMA3 Bootstrapping Window
    //
    // L=577 matches the constant in Lwma3CalculateNextWorkRequired.
    // Blocks at heights 1..578 must have nBits == powLimit.
    // Any deviation means a forged or corrupted header.
    // -------------------------------------------------------------------
    static constexpr int64_t LWMA3_L = 577; // must match L in Lwma3CalculateNextWorkRequired
 
    if (height <= LWMA3_L + 1) { // blocks at heights 1..578
        return (new_nbits == pow_limit_compact);
    }
 
    // -------------------------------------------------------------------
    // Full LWMA3 operation (height >= 579)
    //
    // Upper bound: new_target <= old_target * 6
    //
    // Derived from: solvetime = std::min(6 * T, ...)
    // The maximum possible nextTarget under any circumstances is avgTarget * 6.
    // Since avgTarget ≈ old_target (weighted average of recent N block targets),
    // no legitimate block can have new_target > old_target * 6.
    // -------------------------------------------------------------------
    static constexpr int64_t LWMA3_MAX_TARGET_MULTIPLIER = 6;
 
    arith_uint256 max_permitted_target;
    if (old_target > pow_limit / LWMA3_MAX_TARGET_MULTIPLIER) {
        // old_target * 6 would overflow past pow_limit — cap to pow_limit
        max_permitted_target = pow_limit;
    } else {
        max_permitted_target = old_target * LWMA3_MAX_TARGET_MULTIPLIER;
        if (max_permitted_target > pow_limit) {
            max_permitted_target = pow_limit;
        }
    }
 
    if (new_target > max_permitted_target) {
        return false;
    }
 
    return true;
}

bool CheckProofOfWork(uint256 hash, unsigned int nBits, const Consensus::Params& params)
{
    bool fNegative;
    bool fOverflow;
    arith_uint256 bnTarget;

    bnTarget.SetCompact(nBits, &fNegative, &fOverflow);

    // Check range
    if (fNegative || bnTarget == 0 || fOverflow || bnTarget > UintToArith256(params.powLimit))
        return false;

    // Check proof of work matches claimed amount
    if (UintToArith256(hash) > bnTarget)
        return false;

    return true;
}
