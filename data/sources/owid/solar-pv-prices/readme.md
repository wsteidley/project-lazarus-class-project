# Solar photovoltaic panel prices - Data package

This data package contains the data that powers the chart ["Solar photovoltaic panel prices"](https://ourworldindata.org/grapher/solar-pv-prices?v=1&csvType=full&useColumnShortNames=false) on the Our World in Data website. It was downloaded on July 24, 2026.

### Active Filters

A filtered subset of the full data was downloaded. The following filters were applied:

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For most countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The final column is the data column, which is the time series that powers the chart. If the CSV data is downloaded using the "full data" option, then the column corresponds to the time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data column is transformed depending on the chart type and thus the association with the time series might not be as straightforward.


## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

## Detailed information about the data


## Solar photovoltaic module price
This data is expressed in US dollars per watt, adjusted for inflation.
Last updated: August 22, 2025  
Next update: August 2026  
Date range: 1975–2024  
Unit: constant 2024 US$ per watt  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
IRENA (2025); Nemet (2009); Farmer and Lafond (2016) – with major processing by Our World in Data

#### Full citation
IRENA (2025); Nemet (2009); Farmer and Lafond (2016) – with major processing by Our World in Data. “Solar photovoltaic module price” [dataset]. IRENA, “Renewable Power Generation Costs in 2024”; Nemet, “Interim monitoring of cost dynamics for publicly supported energy technologies”; Farmer and Lafond, “How predictable is technological progress?” [original data].
Source: IRENA (2025), Nemet (2009), Farmer and Lafond (2016) – with major processing by Our World In Data

### What you should know about this data
- Solar photovoltaic module prices refer to the cost of the solar panel itself, and do not include installation or other system components.
- Prices are compiled from three sources: Nemet (2009) for 1975-2003, Farmer & Lafond (2016) for 2004-2009, and IRENA for 2010 onward.
- Due to limited data availability, we use the Global Price Index series reported by IRENA, based on pvXchange benchmark prices for modules sold in Europe.
- Historical prices have been adjusted for inflation using the [World Bank's US GDP deflator](https://data.worldbank.org/indicator/NY.GDP.DEFL.ZS?locations=US).

### Sources

#### IRENA – Renewable Power Generation Costs in 2024
Retrieved on: 2025-08-22  
Retrieved from: https://www.irena.org/Publications/2025/Jun/Renewable-Power-Generation-Costs-in-2024  

#### Nemet – Interim monitoring of cost dynamics for publicly supported energy technologies
Retrieved on: 2023-12-12  
Retrieved from: https://pcdb.santafe.edu/graph.php?curve=158  

#### Farmer and Lafond – How predictable is technological progress?
Retrieved on: 2023-12-12  
Retrieved from: https://www.sciencedirect.com/science/article/pii/S0048733315001699  

#### Notes on our processing step for this indicator
- Historical photovoltaic cost data between 1975 and 2003 has been taken from Nemet (2009), and between 2004 and 2009 from Farmer & Lafond (2016).
- From 2010 onward, prices come from IRENA's Renewable Power Generation Costs report, based on pvXchange benchmarks for modules sold in Europe, using the 'Thin film a-Si/u-Si or Global Price Index (from Q4 2013)' series.
- Prices from Nemet (2009) and Farmer & Lafond (2016) have been converted to 2024 US$ using the US GDP deflator, to account for the effects of inflation. The deflator data is available from the [World Bank World Development Indicators](https://data.worldbank.org/indicator/NY.GDP.DEFL.ZS?locations=US).


    