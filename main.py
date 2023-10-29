import numpy as np


def calculate_initial_story_points(total_stories, stories_with_estimates, story_points_for_estimated_stories):
    avg_story_points = story_points_for_estimated_stories / stories_with_estimates
    extrapolated_points = avg_story_points * (total_stories - stories_with_estimates)
    return story_points_for_estimated_stories + extrapolated_points

def run_simulation(total_initial_points, simulations, burn_rate, total_stories):
    avg_story_size = total_initial_points / total_stories
    simulation_results = {}

    adjusted_points = total_initial_points
    # For each type of impact
    for impact_key, simulation_data in simulations.items():
        total_impact_points = 0

        # Simulating for each story individually
        for _ in range(total_stories):
            event = np.random.binomial(1, simulation_data['probability'])
            impact_points = event * simulation_data['impact'] * avg_story_size
            total_impact_points += impact_points

        adjusted_points += total_impact_points
        simulation_results[impact_key] = total_impact_points

    simulation_results['delivery_weeks'] = adjusted_points / burn_rate
    simulation_results['final_points'] = adjusted_points
    return simulation_results


def monte_carlo_simulation(burn_rate, total_stories, stories_with_estimates,
                           story_points_for_estimated_stories, num_simulations, simulations):
    total_initial_points = calculate_initial_story_points(
        total_stories, stories_with_estimates, story_points_for_estimated_stories)

    all_sim_results = [
        run_simulation(
            total_initial_points,
            simulations,
            burn_rate,
            total_stories
        ) for _ in range(num_simulations)
    ]

    aggregated_data = {
        'delivery_weeks': np.mean([sim['delivery_weeks'] for sim in all_sim_results]),
        'final_points': np.mean([sim['final_points'] for sim in all_sim_results]),
        'total_initial_points': total_initial_points,
        'extrapolated_points': total_initial_points - story_points_for_estimated_stories
    }

    return all_sim_results, aggregated_data


def print_results(simulation_results, aggregated_data):
    print("\nSimulation Results:")
    for key in simulation_results[0]:
        values = [sim[key] for sim in simulation_results]
        print(
            f"Simulation {key.replace('_', ' ').capitalize()}: Mean = {np.mean(values):.2f}, Std = {np.std(values):.2f}")

    print("\nAggregated Data:")
    for key, value in aggregated_data.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value:.2f}")

def main():
    # Variables for the release
    story_total = 633
    story_estimated_count = 421
    story_points_estimated_total = 1796
    burn_rate = 320
    simulation_count = 10000

    simulations = {
        'simulate_estimation_errors': {
            'probability': 0.2,
            'impact': 0.3,
            'description': """
            Represents the occurrence of estimation errors during the project's lifecycle.

            Parameters:
            - probability: Likelihood of an estimation error for any given story. (0.3 means 30% chance).
            - impact: Average percentage by which the initial estimate might be off (0.3 implies 30%).
            - Derived from Jira using the planned versus delivered date
            """
        },

        'simulate_rework': {
            'probability': 0.1059,
            'impact': 0.5,
            'description': """
            Represents scenarios where tasks need revisiting due to reasons like bugs, missed requirements, or feedback.

            Parameters:
            - probability: Probability that rework will be required for a given story (0.11 or 11%).
            - impact: Average percentage of the original story points added due to rework (0.5 or 50%).
            - Derived from LinearB using the rework metric
            """
        },

        'simulate_fat_stories': {
            'probability': 0.0142,
            'impact': 3.0,
            'description': """
            Simulates when some stories are larger than originally estimated ("fat" stories).

            Parameters:
            - probability: Likelihood of encountering a fat story (0.0142 or 1.42%).
            - impact: Average multiple by which the original story points might increase for a fat story (3 times).
            - Derived from Jira from any story more than 5 points
            """
        },

        'simulate_unplanned_work': {
            'probability': 0.16,
            'impact': 2.0,
            'description': """
            Represents unexpected tasks that emerge during the project.

            Parameters:
            - probability: Chance of unplanned work emerging (0.16 or 16%).
            - impact: Average multiple of story points added due to unplanned work (2 times the effort).
            - Derived from Jira using an unplanned_work label
            """
        },

        'simulate_defects': {
            'probability': 0.07,
            'impact': 1.0,
            'description': """
            Simulates the occurrence of defects in the code that need fixing.

            Parameters:
            - probability: Likelihood of a defect in a given story (0.10 or 10%).
            - impact: Average story points added due to fixing defects (equivalent to a regular story).
            - Derived from Jira by summing up points from defects identified during the release
            """
        },

        'simulate_dependencies': {
            'probability': 0.1,
            'impact': 1.0,
            'description': """
            Represents scenarios where a story is dependent on another story causing delays.

            Parameters:
            - probability: Chance that a story has a dependency (0.3 or 30%).
            - impact: Average story points added due to dependencies.
            - Derived from Jira by count the stories that have dependencies on them
            """
        },

        'simulate_blocked_stories': {
            'probability': 0.1,
            'impact': 1.0,
            'description': """
            Simulates when a story cannot progress due to blockers.

            Parameters:
            - probability: Likelihood of a story getting blocked (0.1 or 10%).
            - impact: Average story points added due to resolving blockers.
            - Derived from Jira by looking at the number of blocked points on average 
            """
        }
    }

    sim_results, agg_data = monte_carlo_simulation(
        burn_rate, story_total, story_estimated_count,
        story_points_estimated_total, simulation_count, simulations)

    print_results(sim_results, agg_data)

if __name__ == "__main__":
    main()
