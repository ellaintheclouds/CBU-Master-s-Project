# %% Import Packages and Functions --------------------------------------------------
from datetime import datetime


# %% Define the Function --------------------------------------------------
def sort_data(file_name):
    """Sort data based on species and day number."""
    # Define the species
    species = 'Chimpanzee' if 'C' in file_name else 'Human' if 'H' in file_name else None

    # Chimpanzee ----------
    if species == 'Chimpanzee':

        # Define the day number
        day_number = next((f'Day {i}' for i in range(1, 365) if f'_d{i}_' in file_name), 'Unknown Day')
        
        # Define timepoint based on the day number
        if day_number in ['Day 95', 'Day 96']:
            timepoint = 't1'
        elif day_number == 'Day 153':
            timepoint = 't2'
        elif day_number in ['Day 184', 'Day 185']:
            timepoint = 't3'
        else:
            timepoint = 'Unknown Timepoint'

        # Define the cell line
        cell_line = None  # No cell line information for chimpanzee

    # Human ----------
    elif species == 'Human':
        
        # Extract seeding and recording dates
        try:
            parts = file_name.split('_')

            # Find the seeding date
            seeding_date_str = parts[2]
            seeding_date = datetime.strptime(seeding_date_str, '%Y%m%d')

            # Find the recording date
            recording_date_str = parts[3]
            recording_date = datetime.strptime(recording_date_str, '%Y%m%d')

            # Define the day number
            day_number = f'Day {(recording_date - seeding_date).days}'
        
        except (IndexError, ValueError):
            day_number = 'Unknown Day'

        # Define the timepoint based on the day number
        try:
            day_number_int = int(day_number.split()[1])  # Extract the numeric day value
            if 90 <= day_number_int <= 134:
                timepoint = 't1'
            elif 135 <= day_number_int <= 164:
                timepoint = 't2'
            elif 165 <= day_number_int <= 185:
                timepoint = 't3'
            else:
                timepoint = 'Unknown Timepoint'

        except (IndexError, ValueError):
            timepoint = 'Unknown Timepoint'

        # Define the cell line
        if species == 'Human':
            if 'fiaj' in file_name:
                cell_line = 'fiaj'
            elif 'hehd1' in file_name:
                cell_line = 'hehd1'
            elif 'scti003a' in file_name.lower():
                cell_line = 'SCTi003A'
            elif 'pahc4' in file_name.lower():
                cell_line = 'Pahc4'
            elif 'bioni' in file_name.lower():
                cell_line = 'bioni'
            else:
                cell_line = 'Unknown Cell Line'
        
        else:
            cell_line = None

    return species, day_number, timepoint, cell_line