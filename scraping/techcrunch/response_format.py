climate_sectors='Energy,Transportation,Agriculture,Forestry,Water and Oceans,Built Environment,Waste Management,Land Use and Ecosystem Services,Carbon Markets and Climate Finance,Climate Adaptation and Resilience,Circular Economy,Environmental Technology,Carbon Removal,Climate Advocacy and Policy'
locations='North America, South America, Europe, Russia, Asia, Middle East, Other'

response_format = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "name": "company_info_from_article",
        "schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Name of the company"
                },
                "founders": {
                    "type": ["string", "null"],  # Optional array
                    "items": {
                        "type": "string"
                    },
                    "description": "CSV string of founder names"
                },
                "is_climate_related": {
                    "type": "boolean",
                    "description": "Indicates if the company is climate-related"
                },
                "climate_sectors": {
                    "type": ["string", "null"],  # Optional string
                    "description":f"If is_climate_related is true, this has a list of the climate sectors from {climate_sectors} related to this company",
                },
                "location": {
                    "type": ["string", "null"],
                    "description":f"Location of the company from {locations} list",
                },
                "living_status": {
                    "type": "string",
                    "description": "Indicates if the company is dead, living, or unknown"
                },
                "has_pivoted": {
                    "type": ["boolean", "null"],
                    "description": "Indicates if the company has pivoted from its original idea"
                },
                "year_founded": {
                    "type": ["string", "null"],
                    "description": "Year the company was founded"
                },
                "year_died": {
                    "type": ["string", "null"],
                    "description": "Year the company closed, if applicable"
                },
                "idea_summary": {
                    "type": ["string", "null"],
                    "description": "One-sentence summary of the startup idea and technology"
                },
                "reason_for_demise": {
                    "type": ["string", "null"],
                    "description": "If applicable, describes reasons for the company's demise"
                }
            },
            "required": [
                "company_name", "founders", "is_climate_related", "climate_sectors",
                "location", "living_status", "has_pivoted", "year_founded", "year_died",
                "idea_summary", "reason_for_demise"
            ],
            "additionalProperties": False
        },
    }   
}
response_format_llama = {
    "company_name": {
        "type": "string",
        "description": "Name of the company"
    },
    "founders": {
        "type": ["string", "null"],  # Optional array
        "items": {
            "type": "string"
        },
        "description": "CSV string of founder names"
    },
    "is_climate_related": {
        "type": "boolean",
        "description": "Indicates if the company is climate-related"
    },
    "climate_sectors": {
        "type": ["string", "null"],  # Optional string
        "description":f"If is_climate_related is true, this has a list of the climate sectors from {climate_sectors} related to this company",
    },
    "location": {
        "type": ["string", "null"],
        "description":f"Location of the company from {locations} list",
    },
    "living_status": {
        "type": "string",
        "description": "Indicates if the company is dead, living, or unknown"
    },
    "has_pivoted": {
        "type": ["boolean", "null"],
        "description": "Indicates if the company has pivoted from its original idea"
    },
    "year_founded": {
        "type": ["string", "null"],
        "description": "Year the company was founded"
    },
    "year_died": {
        "type": ["string", "null"],
        "description": "Year the company closed, if applicable"
    },
    "idea_summary": {
        "type": ["string", "null"],
        "description": "One-sentence summary of the startup idea and technology"
    },
    "reason_for_demise": {
        "type": ["string", "null"],
        "description": "If applicable, describes reasons for the company's demise"
    }
}


# These need to be added in another layer otherwise it's too much for one layer
# and the agent gets confused for some fields

funding_rounds='Pre-Seed, Seed, Series A, Series B, Series C, Series D, Series E, Series F, Other'
# Unable to get failure categories to work just yet
# I think there needs to be another layer focused on just this
# failure_categories='Capital Intensity, Market Problems, Financial Issues, Leadership or Team-Related, Product Issues, Competition and Market Entry, Business Model Failures, Operational Issues, Strategic Mistakes, Policy or Regulatory Dependence'
# "failure_category": {
#                     "type": ["string", "null"],
#                     "description":f"If living_status is dead, this has a list of the categories causing the failure from {failure_categories}",
#                 },

response_format_funding_rounds = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "name": "company_funding_info",
        "schema": {
            "type": "object",
            "properties": {
                "funding_rounds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "currency_symbol": {
                                "type": ["string", "null"],
                                "description":"Currency Symbol for the money"
                            },
                            "round_name": {
                                "type": ["string", "null"],
                                "description": f"Type of funding round should be one of {funding_rounds}"
                            },
                            "amount": {
                                "type": ["number", "null"],
                                "description": "Amount of the funding round in integer format"
                            },
                            "date": {
                                "type": ["string", "null"],
                                "description": "String representing MM/YY date of the funding"
                            }
                        },
                        "required": ["round_name", "amount", "date"],
                        "additionalProperties": False
                    }

                }
            },
            "required": [
                "funding_rounds"
            ],
            "additionalProperties": False
        },
    }   
}


response_format_funding_rounds_llama = {
    "funding_rounds": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "currency_symbol": {
                    "type": ["string", "null"], 
                    "description": "Currency symbol for funding round"
                },
                "round_name": {
                    "type": ["string", "null"], 
                    "description": "Should be one of [{funding_rounds}]"
                },
                "amount": {
                    "type": ["number", "null"], 
                    "description": "Number representing the amount of funding raised"
                },
                "date": {
                    "type": ["string", "null"], 
                    "description": "Date in format MM/YY"
                }
            },
            "required": ["round_name", "amount", "date"],
            "additionalProperties": False
        }
    }
}