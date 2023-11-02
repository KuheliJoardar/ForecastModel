import datetime
import os
from jira import JIRA
import numpy as np


def calculate_initial_story_points(total_stories, stories_without_estimates, story_points_for_estimated_stories):
    """
    Calculate the total estimated story points for all stories, both estimated and unestimated.

    The function first computes the average story points for the stories that have been estimated.
    It then extrapolates this average to the unestimated stories and returns the total estimated
    story points for all stories combined.

    Parameters:
    - total_stories (int): The total number of stories (both estimated and unestimated).
    - stories_with_estimates (int): The number of stories that have been estimated.
    - story_points_for_estimated_stories (float): The total story points for the stories that have been estimated.

    Returns:
    - float: The total estimated story points for all stories.

    Example:
    calculate_initial_story_points(100, 50, 250)
    500.0

    Notes:
    - Ensure that the stories_with_estimates is never greater than total_stories.
    - The function assumes that the distribution of story points for unestimated stories is similar
      to the distribution of story points for estimated stories.
    """
    avg_story_points = story_points_for_estimated_stories / total_stories
    extrapolated_points = avg_story_points * stories_without_estimates
    return story_points_for_estimated_stories + extrapolated_points


def run_simulation(total_initial_points, simulations, burn_rate, total_stories):
    """
        Runs a simulation to estimate the total points considering different impacts and calculates the delivery weeks.

        This function simulates the impact of different factors on the total story points. For each type of impact,
        it uses a binomial distribution to determine whether a specific impact event occurs for a story.
        If the event occurs, the story's points are adjusted based on the defined impact multiplier.
        Finally, the function computes the number of delivery weeks required given the adjusted story points and a defined burn rate.

        Parameters:
        - total_initial_points (float): The initial total story points before any impact.
        - simulations (dict): A dictionary where each key represents a type of impact, and the associated value is another dictionary with keys 'probability' (probability of the impact event occurring for a story) and 'impact' (multiplier for story points if the event occurs).
        - burn_rate (float): The number of story points that can be delivered per week.
        - total_stories (int): The total number of stories.

        Returns:
        - dict: A dictionary with keys representing different types of impacts and their corresponding total adjusted points, 'delivery_weeks' (estimated weeks for delivery) and 'final_points' (total adjusted story points after all impacts).

        Example:
        simulations_data = {
            'type1': {'probability': 0.1, 'impact': 0.2},
            'type2': {'probability': 0.05, 'impact': 0.5}
        }
        run_simulation(1000, simulations_data, 50, 100)
        {'type1': ..., 'type2': ..., 'delivery_weeks': ..., 'final_points': ...}

        Notes:
        - The function assumes a binomial distribution for the occurrence of impact events for stories.
        - Ensure the 'probability' values in simulations are between 0 and 1, and the 'impact' values are non-negative.
        """
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


def monte_carlo_simulation(
        burn_rate,
        total_stories,
        stories_without_estimates,
        story_points_for_estimated_stories,
        num_simulations,
        simulations
):
    """
        Conducts a Monte Carlo simulation to estimate delivery weeks and total points based on various impacts.

        This function utilizes the Monte Carlo method to repeatedly run simulations and assess the impact of various factors on
        the total story points and consequently, the delivery weeks. It aggregates the results to provide an average estimate.

        Parameters:
        - burn_rate (float): The number of story points that can be delivered per week.
        - total_stories (int): The total number of stories.
        - stories_with_estimates (int): The number of stories that have been estimated.
        - story_points_for_estimated_stories (float): The total story points for stories that have been estimated.
        - num_simulations (int): The number of simulations to run.
        - simulations (dict): A dictionary where each key represents a type of impact, and the associated value is another dictionary with keys 'probability' (probability of the impact event occurring for a story) and 'impact' (multiplier for story points if the event occurs).

        Returns:
        - tuple: A tuple where the first element is a list containing the result of each simulation, and the second element is a dictionary aggregating the results (average 'delivery_weeks', 'final_points', 'total_initial_points' and 'extrapolated_points').

        Example:
        simulations_data = {
            'type1': {'probability': 0.1, 'impact': 0.2},
            'type2': {'probability': 0.05, 'impact': 0.5}
        }
        monte_carlo_simulation(50, 100, 50, 500, 1000, simulations_data)
        ([...], {'delivery_weeks': ..., 'final_points': ..., 'total_initial_points': ..., 'extrapolated_points': ...})

        Notes:
        - The function leverages the Monte Carlo method which involves running the simulation multiple times to estimate the outcomes.
        - Ensure the 'probability' values in simulations are between 0 and 1, and the 'impact' values are non-negative.
        """

    total_initial_points = calculate_initial_story_points(
        total_stories, stories_without_estimates, story_points_for_estimated_stories)

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


def get_delivery_date(start_date, delivery_weeks):
    """
    Calculate the estimated delivery date in business days.

    Parameters:
    - start_date (str): The starting date in 'YYYY-MM-DD' format.
    - delivery_weeks (float): The number of weeks to deliver.

    Returns:
    - str: Estimated delivery date in 'YYYY-MM-DD' format.
    """
    # Convert weeks to business days. Assuming 5 business days in a week.
    business_days = int(delivery_weeks * 5)

    # Calculate the delivery date
    delivery_date = np.busday_offset(start_date, business_days, roll='forward')

    return str(delivery_date)


def print_results(start_date, simulation_results, aggregated_data):
    print("\nSimulation Results:")

    # Calculate mean impacts and sort keys by these values in descending order
    sorted_keys = sorted(simulation_results[0], key=lambda k: -np.mean([sim[k] for sim in simulation_results]))

    for key in sorted_keys:
        values = [sim[key] for sim in simulation_results]
        print(
            f"Simulation {key.replace('_', ' ').capitalize()}: Mean = {np.mean(values):.2f}, Std = {np.std(values):.2f}")

    print("\nAggregated Data:")
    for key, value in aggregated_data.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value:.2f}")

    print(f"Delivery date {get_delivery_date(start_date, aggregated_data['delivery_weeks'])}")


def bottleneck_probability(num_engineers, num_stories):
    # Calculate the average stories per engineer
    avg_stories_per_engineer = num_stories // num_engineers

    # Calculate the number of stories that would be evenly distributed without causing a bottleneck
    non_bottleneck_stories = avg_stories_per_engineer * num_engineers

    # Calculate the number of stories causing bottlenecks
    bottleneck_stories = num_stories - non_bottleneck_stories

    # Probability that a randomly chosen story will be a bottleneck-causing story
    probability = bottleneck_stories / num_stories

    return probability


def fetch_and_analyze_stories(jira_url, jql_query):
    api_key = os.environ.get('JIRA_ACCESS_TOKEN')
    if not api_key:
        raise ValueError("JIRA_ACCESS_TOKEN environment variable not set.")

    user_name = os.environ.get('JIRA_USER_NAME')
    if not api_key:
        raise ValueError("JIRA_USER_NAME environment variable not set.")

    options = {
        "server": jira_url
    }

    jira = JIRA(options, basic_auth=(user_name, api_key))

    # Fetch issues using the provided JQL query
    issues = jira.search_issues(jql_query, maxResults=1000)  # Adjust maxResults as needed

    # Analyze the issues. For the sake of this example, I'm only counting stories.
    # Extend this section as needed based on the metrics you need to extract.

    story_total = len(issues)
    story_points_estimated_total = sum([issue.fields.customfield_10026 for issue in issues if issue.fields.customfield_10026])
    stories_without_estimates = sum(1 for issue in issues if not getattr(issue.fields, 'customfield_10026', None))

    # Return analysis results
    return {
        'story_total': story_total,
        'story_points_estimated_total': story_points_estimated_total,
        'stories_without_estimates':stories_without_estimates
    }


def construct_jql_query(project, status_not_in, component_not_in, fix_version_in, sprint=None):
    """
    Constructs a JQL query based on the provided parameters.

    Args:
        project (str): Project name in JIRA.
        status_not_in (list): List of statuses to exclude.
        component_not_in (list): List of components to exclude.
        fix_version_in (list): List of fix versions to include.
        sprint (int, optional): Sprint number. Defaults to None.

    Returns:
        str: JQL query string.
    """
    # Constructing the base JQL
    jql = f'project = "{project}"'

    # Adding status exclusions
    status_exclusions = ', '.join([f'"{status}"' for status in status_not_in])
    jql += f' AND status not in ({status_exclusions})'

    # Adding component exclusions
    component_exclusions = ', '.join([f'"{component}"' for component in component_not_in])
    jql += f' AND (component not in ({component_exclusions}) OR component is EMPTY)'

    # Adding fix versions inclusions
    fix_versions = ', '.join([f'"{version}"' for version in fix_version_in])
    jql += f' AND fixVersion in ({fix_versions})'

    # Adding sprint filter if provided
    if sprint:
        jql += f' AND Sprint = {sprint}'

    return jql

def main():

    jql_query = construct_jql_query(
        project="Layr Platform",
        status_not_in=["Done", "Abandoned", "Ready for Production", "Verify"],
        component_not_in=["Parent"],
        fix_version_in=["Alpha - 1.4.0"],
    )

    jira_data = fetch_and_analyze_stories(
        jira_url=r'https://teamlayr.atlassian.net/',
        jql_query=jql_query
    )

    story_total = jira_data['story_total']
    stories_without_estimates = jira_data['stories_without_estimates']
    story_points_estimated_total = jira_data['story_points_estimated_total']
    burn_rate = 280
    num_engineers = 18
    simulation_count = 10000

    probability_of_bottleneck = bottleneck_probability(num_engineers, story_total)

    simulations = {
        'estimation_errors': {
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

        'discovered_scope': {
            'probability': 0.65,
            'impact': 1,
            'description': """
        Represents the occurrence of additional engineering scope in the development of a story.

        Parameters:
        - probability: Liklihood that while working on a story we discover more engineering scope.
        - impact: The average number of stories created when engineering scope is discovered
        - Derived from Jira identifying issues created after the start of the release
        """
        },


        'rework': {
            'probability': 0.1059,
            'impact': 0.6,
            'description': """
            Represents scenarios where tasks need revisiting due to reasons like bugs, missed requirements, or feedback.

            Parameters:
            - probability: Probability that rework will be required for a given story (0.11 or 11%).
            - impact: Average percentage of the original story points added due to rework (0.5 or 50%).
            - Derived from LinearB using the rework metric
            """
        },

        'refactor': {
            'probability': 0.1641,
            'impact': 0.6,
            'description': """
        Represents scenarios where tasks lead to refactoring to clear technical debt.

        Parameters:
        - probability: Probability that refactoring will be required for a given story.
        - impact: Average percentage of the original story points added due to refactoring.
        - Derived from LinearB using the rework metric
        """
        },

        'fat_stories': {
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

        'unplanned_work': {
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

        'defects': {
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

        'dependencies': {
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

        'bottlenecks': {
            'probability': probability_of_bottleneck,
            'impact': 1.0,
            'description': """
        Represents scenarios where a number of stories are dependent on one person - a bottleneck.

        Parameters:
        - probability: Chance that one engineer is assigned many stories forcing serial development.
        - impact: Average story points added due to the bottleneck.
        - Derived from Jira by count the stories that the same engineer assigned
        """
        },

        'automated_testing': {
            'probability': 0.85,
            'impact': 1,  # This is a mix of automated versus manual work
            'description': """
         Represents scenarios where a story may require automated testing work

         Parameters:
         - probability: Chance that a story will require automated testing.
         - impact: Average story points added due to dependencies.
         - Derived from Jira by looking at stories that have a automated testing label
         """
        },

        'fix_broken_test': {
            'probability': 0.05,
            'impact': 0.7,  # This is a mix of automated versus manual work
            'description': """
         Represents scenarios where a story breaks an existing test
    
         Parameters:
         - probability: Chance that a story will break an existing test
         - impact: Average amount of work required to fix the test
         - Derived from Jira by looking at stories that have a automated testing label
         """
        },

        'manual_testing': {
            'probability': 0.15,
            'impact': 0.25,  # This is a mix of automated versus manual work
            'description': """
         Represents scenarios where a story may require quality engineering work
    
         Parameters:
         - probability: Chance that a story will require manual testing.
         - impact: Average story points added due to manual testing.
         - Derived from Jira by looking at stories that have a manual testing label
         """
        },

        'blocked_stories': {
            'probability': 0.02,
            'impact': 1.0,
            'description': """
            Simulates when a story cannot progress due to blockers.

            Parameters:
            - probability: Likelihood of a story getting blocked (0.1 or 10%).
            - impact: Average story points added due to resolving blockers.
            - Derived from Jira by looking at the number of blocked points on average 
            """
        },

        'merge_issues': {
            'probability': 0.4,
            'impact': 0.5,
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
        burn_rate, story_total, stories_without_estimates,
        story_points_estimated_total, simulation_count, simulations)

    print_results(datetime.date.today(), sim_results, agg_data)


if __name__ == "__main__":
    main()
