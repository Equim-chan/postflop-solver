use dashmap::DashMap;
use fastrand::Rng;
use foldhash::HashMap;
use foldhash::fast::RandomState;
use rayon::prelude::*;

const NUM_ITERS: usize = 80000;

#[derive(Clone, Copy)]
struct ICMEquity {
    short_stack_player: f64,
    deep_stack_player: f64,
}

pub struct ICMCalculator {
    // Payout structure
    payouts: Vec<f64>,
    // Stack list of other players, remains constant across multiple calculations
    other_players_stacks: Vec<f64>,
    // Top-level cache: (Player A's stack) -> (Equity of A and B)
    // Since the total stacks of A and B remain unchanged, using A's stack as key uniquely determines the state
    calculation_cache: DashMap<i32, ICMEquity, RandomState>,
}

impl ICMCalculator {
    /// Create a new ICM calculator instance
    ///
    /// # Arguments
    ///
    /// * `other_players_stacks` - A `Vec<i32>` containing the stacks of all players except A and B.
    /// * `payout_structure` - A `Vec<i32>` containing the payout distribution starting from first place.
    pub fn new(other_players_stacks: Vec<i32>, payout_structure: Vec<i32>) -> Self {
        // Convert payouts to f64 for calculation convenience
        let other_players_stacks = other_players_stacks.into_iter().map(|p| p as f64).collect();
        let payouts = payout_structure.into_iter().map(|p| p as f64).collect();
        Self {
            payouts,
            other_players_stacks,
            calculation_cache: DashMap::default(),
        }
    }

    /// Calculate ICM equity for players A and B given their stacks.
    pub fn calculate(&self, stacks_a: i32, stacks_b: i32) -> (f64, f64) {
        let cache_key = stacks_a.min(stacks_b);
        if let Some(equity) = self.calculation_cache.get(&cache_key) {
            if stacks_a <= stacks_b {
                return (equity.short_stack_player, equity.deep_stack_player);
            } else {
                return (equity.deep_stack_player, equity.short_stack_player);
            }
        }

        if self.payouts.is_empty() {
            let result = ICMEquity {
                short_stack_player: 0.0,
                deep_stack_player: 0.0,
            };
            self.calculation_cache.insert(stacks_a, result);
            return (result.short_stack_player, result.deep_stack_player);
        }

        let num_players = self.other_players_stacks.len() + 2;
        let mut all_stacks = Vec::with_capacity(num_players);
        all_stacks.push(stacks_a as f64);
        all_stacks.push(stacks_b as f64);
        all_stacks.extend_from_slice(&self.other_players_stacks);

        let all_equities = if self.payouts.len() > 16 || num_players > 64 {
            self.calculate_estimate(&all_stacks, NUM_ITERS)
        } else {
            // Call internal recursive function for complete calculation
            // Create a new memo for this complete calculation
            let mut memo = HashMap::default();
            // Initial bitmask, all bits are 1, indicating all players participate
            let initial_mask = u64::MAX >> (64 - num_players);
            self.calculate_exact_recursive(&all_stacks, initial_mask, 0, &mut memo)
        };

        // Extract results for A and B
        let equities_a = all_equities[0];
        let equities_b = all_equities[1];
        let (short_stack_player, deep_stack_player) = if stacks_a <= stacks_b {
            (equities_a, equities_b)
        } else {
            (equities_b, equities_a)
        };
        let result = ICMEquity {
            short_stack_player,
            deep_stack_player,
        };

        // Store in top-level cache and return
        self.calculation_cache.insert(cache_key, result);
        (equities_a, equities_b)
    }

    /// Internal recursive function that computes ICM using bitmask and memoization
    ///
    /// # Arguments
    ///
    /// * `all_stacks` - List of all players' stacks (f64)
    /// * `player_mask` - A bitmask representing players currently participating in the calculation
    /// * `payout_idx` - Index of the current payout being distributed in `self.payouts`
    /// * `memo` - Memoization table for storing subproblem results
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` whose element order corresponds to the player order of bits set to 1 in `player_mask`.
    fn calculate_exact_recursive(
        &self,
        all_stacks: &[f64],
        player_mask: u64,
        payout_idx: usize,
        memo: &mut HashMap<(u64, usize), Vec<f64>>,
    ) -> Vec<f64> {
        // Check if result already exists in memo
        if let Some(cached_result) = memo.get(&(player_mask, payout_idx)) {
            return cached_result.clone();
        }

        // Base case for recursion: no more payouts to distribute, or no players left
        let num_active_players = player_mask.count_ones() as usize;

        // Initialize equity vector for current active players
        let mut total_equities = vec![0.0; num_active_players];

        if payout_idx >= self.payouts.len() || num_active_players == 0 {
            return total_equities;
        }

        // Extract stacks and indices of current active players
        let mut active_player_indices = Vec::with_capacity(num_active_players);
        let mut sub_total_stacks = 0.0;
        for (i, stacks) in all_stacks.iter().enumerate() {
            if (player_mask >> i) & 1 == 1 {
                active_player_indices.push(i);
                sub_total_stacks += stacks;
            }
        }

        // If remaining players' total stack is 0, their equity is also 0
        if sub_total_stacks == 0.0 {
            return total_equities;
        }

        // Recursive core: iterate through each active player, calculate their probability of winning current position, and recursively calculate remaining players' equity
        for i in 0..num_active_players {
            let winner_original_idx = active_player_indices[i];
            let winner_stacks = all_stacks[winner_original_idx];

            // Probability of this player winning the current position
            let prob_win = winner_stacks / sub_total_stacks;

            // Add equity directly obtained from winning current position
            total_equities[i] += prob_win * self.payouts[payout_idx];

            // If there's next level payout, recursively calculate
            let next_mask = player_mask & !(1u64 << winner_original_idx);
            if next_mask != 0 && payout_idx + 1 < self.payouts.len() {
                // Recursive call to calculate equity distribution of remaining players if current player wins
                let sub_equities =
                    self.calculate_exact_recursive(all_stacks, next_mask, payout_idx + 1, memo);

                // Distribute subproblem equity to other players with probability weighting
                let mut sub_equity_idx = 0;
                for (j, total_equity) in total_equities
                    .iter_mut()
                    .enumerate()
                    .take(num_active_players)
                {
                    if i == j {
                        // Skip current round winner
                        continue;
                    }
                    *total_equity += prob_win * sub_equities[sub_equity_idx];
                    sub_equity_idx += 1;
                }
            }
        }

        // Store calculation result in memo and return
        memo.insert((player_mask, payout_idx), total_equities.clone());
        total_equities
    }

    fn calculate_estimate(&self, chip_stacks: &[f64], num_iters: usize) -> Vec<f64> {
        let num_players = chip_stacks.len();
        let num_payouts = self.payouts.len();

        // Calculate exponents
        let total_chips: f64 = chip_stacks.iter().sum();
        let avg_chips = total_chips / num_players as f64;
        let exponents: Vec<f32> = chip_stacks
            .iter()
            .map(|&stack| (avg_chips / stack) as f32)
            .collect();

        let total_iters = num_iters * num_players;
        let num_threads = rayon::current_num_threads();
        let iters_per_thread = total_iters / num_threads + 1;
        (0..num_threads)
            .into_par_iter()
            .map(|_| {
                let mut rng = Rng::new();
                let mut indexed_values: Vec<(usize, f32)> =
                    (0..num_players).map(|i| (i, 0.0)).collect();
                let mut equities = vec![0.0; num_players];
                for _ in 0..iters_per_thread {
                    // Generate random values with exponents
                    for (id, (v, &exp)) in indexed_values.iter_mut().zip(&exponents).enumerate() {
                        *v = (id, rng.f32().powf(exp));
                    }

                    // Only sort the top-k elements needed for payouts
                    if num_payouts < num_players {
                        indexed_values.select_nth_unstable_by(num_payouts, |a, b| {
                            b.1.partial_cmp(&a.1).unwrap()
                        });
                        indexed_values[..num_payouts]
                            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    } else {
                        indexed_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    }

                    // Distribute payouts to top finishers
                    for (&(player_id, _), &payout) in indexed_values.iter().zip(&self.payouts) {
                        equities[player_id] += payout;
                    }
                }
                for equity in &mut equities {
                    *equity /= iters_per_thread as f64;
                }
                equities
            })
            .reduce(
                || vec![0.0; num_players],
                |a, b| {
                    let (mut a, b) = if a.capacity() >= b.capacity() {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    a.iter_mut()
                        .zip(b)
                        .for_each(|(a, b)| *a += b / num_threads as f64);
                    a
                },
            )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic() {
        let other_players_stacks = vec![1000; 14];
        let payout_structure = vec![50, 30, 20];
        let calculator = ICMCalculator::new(other_players_stacks, payout_structure);

        let (equity_a, equity_b) = calculator.calculate(1000, 1000);

        // With 16 players having equal stacks, their equity should be the same.
        // Total payout is 50 + 30 + 20 = 100.
        // Equity per player is 100 / 16 = 6.25.
        assert!((equity_a - 6.25).abs() < 1e-9);
        assert!((equity_b - 6.25).abs() < 1e-9);

        // Test with different stacks
        let (equity_a, equity_b) = calculator.calculate(2000, 500);
        // Player A with a larger stack should have higher equity.
        assert!(equity_a > equity_b);
        // The sum of equities for A and B should be roughly 2 * 6.25 = 12.5,
        // but not exactly, due to ICM pressure.
        // Let's just check they are reasonable values.
        assert!(equity_a > 6.25);
        assert!(equity_b < 6.25);
    }

    #[test]
    fn player_5_payouts_3() {
        let other_players_stacks = (1..9).collect();
        let payout_structure = vec![50, 30, 20];
        let calculator = ICMCalculator::new(other_players_stacks, payout_structure);

        let (equity_a, equity_b) = calculator.calculate(9, 10);
        assert!((equity_a - 15.794621704108263).abs() <= f64::EPSILON);
        assert!((equity_b - 17.216638033941944).abs() <= f64::EPSILON);
    }

    #[test]
    fn player_16_payouts_16() {
        let other_players_stacks = vec![1000; 14];
        let mut payout_structure = vec![50, 30, 20, 10, 5, 2];
        payout_structure.extend_from_slice(&[1; 10]);
        let calculator = ICMCalculator::new(other_players_stacks, payout_structure);

        let (equity_a_0, equity_b_0) = calculator.calculate(800, 1200);
        let (equity_a_1, equity_b_1) = calculator.calculate(800, 1200);
        assert_eq!(equity_a_0, equity_a_1);
        assert_eq!(equity_b_0, equity_b_1);
    }

    #[test]
    fn player_64_payouts_4() {
        let other_players_stacks = vec![1000; 62];
        let payout_structure = vec![50, 30, 20, 10];
        let calculator = ICMCalculator::new(other_players_stacks, payout_structure);

        let (equity_a, equity_b) = calculator.calculate(800, 1200);
        eprintln!("{equity_a}, {equity_b}");
    }

    #[test]
    fn estimate() {
        let other_players_stacks = vec![1000; 100];
        let mut payout_structure = vec![200, 100, 80, 50, 30, 20, 10, 5, 2];
        payout_structure.extend_from_slice(&[1; 20]);
        let calculator = ICMCalculator::new(other_players_stacks, payout_structure);

        let (equity_a, equity_b) = calculator.calculate(800, 1200);
        eprintln!("{equity_a}, {equity_b}");
    }
}
