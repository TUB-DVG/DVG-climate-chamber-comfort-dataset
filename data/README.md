# Climate Chamber Thermal Comfort Dataset - TUB-DVG group

## Overview
This folder contains data collected during a thermal comfort study at the climate chamber of RWTH Aachen University, Germany. The study, conducted between November 2023 and January 2024, aimed to collect user subjective responses to different room climate conditions. This dataset is available for research and analysis in indoor environmental quality, thermal comfort, and related fields.

## Data Collection Process
Data was collected under controlled climate chamber conditions. Participants from diverse demographic groups were exposed to varying indoor conditions, such as temperature, humidity. During the study, participants answered a series of questions regarding their comfort levels, thermal preferences, and other relevant conditions at 10-minute interval. These questions aimed to capture subjective comfort perceptions alongside the environmental metrics recorded in the climate chamber. The questionaire used to collect subjective responses is available [here](SubjectiveQs.pdf) for reference.

## Dataset Description
The [raw datasets](dvg_climate_chamber_exp_raw_dataset) contains data collected from multiple devices and surveys throughout the experiment. Each file in the folder represents a different data source, ranging from environmental conditions to participant feedback and physiological data.

- **device_data.csv**: Contains information on the measurement devices used during the study.
- **elsys.csv**: Records room temperature and humidity data from ELSYS sensors at different height above ground level.
- **experiment_data.csv**: Lists the specific dates on which each participant took part in the study.
- **feedback.csv**: Subjective feedback from participants, collected at 10-minute intervals with timestamps.
- **heartrate.csv**: Contains heart rate data collected from Fitbit devices.
- **ibutton_ankle.csv**: Ankle skin temperature data from iButton sensors.
- **ibutton_wrist.csv**: Wrist skin temperature data from iButton sensors.
- **testo.csv**: Includes measurements of CO₂, temperature, and humidity from Testo sensors.
- **user_data.csv**: Contains demographic and physical information of participants.
- **weather.csv**: Outdoor weather conditions during the study period.

The [final dataset](dvg_climate_chamber_final_data.csv) contains 1,502 samples with 42 columns representing various user attributes and environmental conditions. Below is a summary of key variables. A detailed description of the columns is provided in the [Data Dictionary](data_dictionary.csv).

- **lastdata**: Timestamp of each observation (YYYY-MM-DD HH:MM:SS).
- **therm_sens**: Thermal sensation vote, on a scale from cold (1) to hot (7).
- **therm_comfort**: Thermal comfort vote, from very uncomfortable (1) to very comfortable (6).
- **therm_pref**: Thermal preference (1 = cooler, 2 = no change, 3 = warmer).
- **met**: Activity level (MET) of the partiipants in the last 10 minutes.
- **clo**: Clothing Value (CLO) of the participants in the last 10 minutes. 
- **Environmental and physiological metrics**: Various environmental parameters (e.g., room temperature, humidity, CO₂, air velocity) and physiological metrics (e.g., heart rate, wrist and ankle skin temperatures).

The [processed dataset](processed_data.csv) contains the data cleaned and queried for the ML model development.

### Data Format
The data is provided in a CSV format, where each row represents a single observation. The variables value are averaged over the 10 minutes prior to the timestamp.


## Contact
For questions or further information, please reach out to Julianah Odeyemi at j.odeyemi@tu-berlin.de.


