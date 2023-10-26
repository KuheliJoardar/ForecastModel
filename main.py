import numpy as np


def monte_carlo_simulation(burn_rate, total_stories, stories_with_estimates, defect_points_per_story_point,
                           fat_story_percentage, average_story_points_for_estimated_stories,
                           average_points_for_stories_wo_estimates, dependency_delay_factor,
                           fat_story_multiplier, planned_days_off, num_engineers, unplanned_off_probability,
                           disruption_probability, average_disruption_delay, num_simulations=10000):
    """
    Monte Carlo simulation to predict the delivery date and defects using numpy.

    Args:
    - burn_rate (float): Average story points completed per day.
    - total_stories (int): Total number of stories in the project.
    - stories_with_estimates (int): Number of stories that have point estimates.
    - defect_points_per_story_point (float): Expected defects per story point.
    - fat_story_percentage (float): Percentage of stories that are 'fat'.
    - average_story_points_for_estimated_stories (float): Average story points for stories with estimates.
    - average_points_for_stories_wo_estimates (float): Average story points for stories without estimates.
    - dependency_delay_factor (float): Expected delay due to dependencies.
    - fat_story_multiplier (int): Multiplier for fat stories.
    - planned_days_off (int): Total number of planned days off across all engineers.
    - num_engineers (int): Total number of engineers in the team.
    - unplanned_off_probability (float): Probability of an engineer taking an unplanned day off.
    - disruption_probability (float): Probability of a disruption occurring.
    - average_disruption_delay (float): Average delay in weeks introduced by a disruption.
    - num_simulations (int, optional): Number of simulations to run. Defaults to 10,000.

    Returns:
    - tuple: Mean predicted defects and mean predicted delivery weeks.
    """

    # Determine which stories have estimates
    has_estimate = np.random.rand(num_simulations, total_stories) < (stories_with_estimates / total_stories)

    # Determine which are fat stories
    is_fat = np.random.rand(num_simulations, total_stories) < fat_story_percentage

    # Calculate story points based on whether they have estimates and if they are fat
    story_points = np.where(has_estimate, average_story_points_for_estimated_stories,
                            average_points_for_stories_wo_estimates)
    story_points[is_fat] *= fat_story_multiplier

    # Total up the story points for each simulation
    total_story_points = story_points.sum(axis=1)

    # Simulate disruptions
    disruptions = np.random.rand(num_simulations) < disruption_probability
    disruption_delays = np.where(disruptions, np.random.exponential(scale=average_disruption_delay), 0)

    # Calculate potential days of work for the given period of simulations
    total_engineer_days = num_engineers * num_simulations

    # Deducting planned days off
    days_off_assigned = np.random.choice(total_engineer_days, planned_days_off, replace=False)

    # Simulating unplanned days off
    random_vals = np.random.rand(total_engineer_days)
    unplanned_days_off = np.sum(random_vals < unplanned_off_probability)

    # Adjust the total engineer-days and burn rate
    adjusted_engineer_days = total_engineer_days - len(np.unique(days_off_assigned)) - unplanned_days_off
    adjusted_burn_rate = (adjusted_engineer_days / total_engineer_days) * burn_rate

    # Calculate defects for each simulation
    defects = total_story_points * defect_points_per_story_point

    # Calculate weeks required, incorporating both productivity and dependency delays
    weeks_required = total_story_points / adjusted_burn_rate
    dependency_delays = np.random.exponential(scale=dependency_delay_factor, size=num_simulations)
    weeks_required += dependency_delays + disruption_delays

    return defects.mean(), weeks_required.mean()


def main():
    # ======================
    # Variables Specific to Each Release
    # ======================

    # Sprint Data
    total_stories = 633
    stories_with_estimates = 421
    story_points_for_estimated_stories = 1796
    points_for_stories_wo_estimates = 699

    # Story Characteristics
    fat_story_multiplier = 2
    fat_story_percentage = 0.0142

    # Productivity and Days Off
    planned_days_off = 5  # total number of days off over the simulation

    # ======================
    # Variables Based on Historical Data
    # ======================

    # Productivity and Burn Rates
    burn_rate = 320  # rolling or weighted average based on recent sprints
    num_engineers = 18
    unplanned_off_probability = 0.05  # derived from past data on unplanned offs

    # Quality Metrics
    defect_points_per_story_point = 0.05  # based on historical defect rates

    # Dependencies
    dependency_delay_factor = 0.5  # based on historical dependency delays

    # Disruption Parameters
    disruption_probability = 0.1  # based on past occurrences of disruptions
    average_disruption_delay = 0.5  # average delay due to disruptions historically

    # Compute averages outside of the Monte Carlo function
    average_story_points_for_estimated_stories = story_points_for_estimated_stories / stories_with_estimates
    average_points_for_stories_wo_estimates = points_for_stories_wo_estimates / (total_stories - stories_with_estimates)

    # Running the Monte Carlo simulation
    num_simulations = 10000
    mean_defects, mean_delivery_weeks = monte_carlo_simulation(
        burn_rate, total_stories, stories_with_estimates, defect_points_per_story_point,
        fat_story_percentage, average_story_points_for_estimated_stories,
        average_points_for_stories_wo_estimates, dependency_delay_factor, fat_story_multiplier,
        planned_days_off, num_engineers, unplanned_off_probability, disruption_probability,
        average_disruption_delay, num_simulations)

    print(f"Based on {num_simulations} simulations:")
    print(f"Mean predicted defects: {mean_defects:.2f}")
    print(f"Mean predicted delivery weeks: {mean_delivery_weeks:.2f}")


if __name__ == "__main__":
    main()
