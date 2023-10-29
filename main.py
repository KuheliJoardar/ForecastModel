import numpy as np

import numpy as np


def monte_carlo_simulation(burn_rate, total_stories, stories_with_estimates,
                           story_points_for_estimated_stories, num_engineers, num_simulations, simulations):
    def calculate_initial_story_points():
        avg_story_points = story_points_for_estimated_stories / stories_with_estimates
        estimated_points_no_estimate = avg_story_points * (total_stories - stories_with_estimates)
        return story_points_for_estimated_stories + estimated_points_no_estimate

    def simulate_event_occurrence(num_events, probability):
        return np.random.binomial(1, probability, int(num_events))

    def simulate_impact(events_occurred, impact_multiplier, max_value=None):
        impacts = events_occurred * impact_multiplier
        if max_value:
            impacts = np.clip(impacts, 0, max_value)
        return impacts

    total_initial_points = calculate_initial_story_points()

    simulation_results = {
        key: np.zeros(num_simulations) for key in simulations.keys()
    }
    simulation_results.update({'delivery_weeks': np.zeros(num_simulations), 'final_points': np.zeros(num_simulations)})

    for i in range(num_simulations):
        adjusted_points = total_initial_points

        for impact_key, simulation_data in simulations.items():
            event_count = total_initial_points if "points" in impact_key else total_stories
            events = simulate_event_occurrence(event_count, simulation_data['probability'])
            impact_points = np.sum(simulate_impact(events, simulation_data['impact']))
            adjusted_points += impact_points
            simulation_results[impact_key][i] = impact_points

        simulation_results['delivery_weeks'][i] = adjusted_points / burn_rate
        simulation_results['final_points'][i] = adjusted_points

    aggregated_data = {
        'delivery_weeks': np.mean(simulation_results['delivery_weeks']),
        'final_points': np.mean(simulation_results['final_points']),
        'total_initial_points': total_initial_points,
        'extrapolated_points': total_initial_points - story_points_for_estimated_stories
    }

    return simulation_results, aggregated_data



def main():
    # Variables for the release
    story_total = 633
    story_estimated_count = 421
    story_points_estimated_total = 1796
    burn_rate = 320
    engineer_count = 18
    simulation_count = 10000

    simulations = {
        'simulate_estimation_errors': {
            'probability': 0.3,
            'impact': 0.3,
            '__doc__': """
            Represents the occurrence of estimation errors during the project's lifecycle.

            Parameters:
            - probability: Likelihood of an estimation error for any given story. (0.3 means 30% chance).
            - impact: Average percentage by which the initial estimate might be off (0.3 implies 30%).
            """
        },

        'simulate_rework': {
            'probability': 0.11,
            'impact': 0.5,
            '__doc__': """
            Represents scenarios where tasks need revisiting due to reasons like bugs, missed requirements, or feedback.

            Parameters:
            - probability: Probability that rework will be required for a given story (0.11 or 11%).
            - impact: Average percentage of the original story points added due to rework (0.5 or 50%).
            """
        },

        'simulate_fat_stories': {
            'probability': 0.0142,
            'impact': 3,
            '__doc__': """
            Simulates when some stories are larger than originally estimated ("fat" stories).

            Parameters:
            - probability: Likelihood of encountering a fat story (0.0142 or 1.42%).
            - impact: Average multiple by which the original story points might increase for a fat story (3 times).
            """
        },

        'simulate_unplanned_work': {
            'probability': 0.16,
            'impact': 2,
            '__doc__': """
            Represents unexpected tasks that emerge during the project.

            Parameters:
            - probability: Chance of unplanned work emerging (0.16 or 16%).
            - impact: Average multiple of story points added due to unplanned work (2 times the effort).
            """
        },

        'simulate_defects': {
            'probability': 0.10,
            'impact': 1,
            '__doc__': """
            Simulates the occurrence of defects in the code that need fixing.

            Parameters:
            - probability: Likelihood of a defect in a given story (0.10 or 10%).
            - impact: Average story points added due to fixing defects (equivalent to a regular story).
            """
        },

        'simulate_dependencies': {
            'probability': 0.3,
            'impact': 1,
            '__doc__': """
            Represents scenarios where a story is dependent on another story causing delays.

            Parameters:
            - probability: Chance that a story has a dependency (0.3 or 30%).
            - impact: Average story points added due to dependencies.
            """
        },

        'simulate_blocked_stories': {
            'probability': 0.1,
            'impact': 1,
            '__doc__': """
            Simulates when a story cannot progress due to blockers.

            Parameters:
            - probability: Likelihood of a story getting blocked (0.1 or 10%).
            - impact: Average story points added due to resolving blockers.
            """
        }
    }

    simulation_results, aggregated_data = monte_carlo_simulation(
        burn_rate, story_total, story_estimated_count,
        story_points_estimated_total, engineer_count, simulation_count, simulations)

    # Printing simulation results
    print("\nSimulation Results:")
    for key, value in simulation_results.items():
        print(f"Simulation {key.replace('_', ' ').capitalize()}: Mean = {np.mean(value):.2f}, Std = {np.std(value):.2f}")

    # Print the aggregated data
    print("\nAggregated Data:")
    for key, value in aggregated_data.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value:.2f}")

if __name__ == "__main__":
    main()
