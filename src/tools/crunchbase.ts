import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import { requireCrunchbaseKey } from '../config.js'

const crunchbaseFieldIds = [
  'created_at',
  'entity_def_id',
  'facebook',
  'facet_ids',
  'identifier',
  'image_id',
  'image_url',
  'linkedin',
  'location_identifiers',
  'name',
  'permalink',
  'short_description',
  'stock_exchange_symbol',
  'twitter',
  'updated_at',
  'uuid',
  'website_url',
]

// Fetches Crunchbase Basic API data to help fill in company details. Ported from
// company_name_crunchbase_search in tools_search.py — the Python version called
// response.text() (a string, not callable), which this fetch-based version fixes.
export const crunchbaseCompanySearch = tool(
  async ({ companyName }: { companyName: string }): Promise<string> => {
    try {
      const apiKey = requireCrunchbaseKey()
      const payload = {
        field_ids: crunchbaseFieldIds,
        query: [
          {
            operator_id: 'includes',
            type: 'predicate',
            field_id: 'facet_ids',
            values: ['company'],
          },
          {
            operator_id: 'eq',
            type: 'predicate',
            field_id: 'identifier',
            values: [companyName],
          },
        ],
      }

      const response = await fetch('https://api.crunchbase.com/v4/data/searches/organizations', {
        method: 'POST',
        headers: {
          accept: 'application/json',
          'content-type': 'application/json',
          'X-cb-user-key': apiKey,
        },
        body: JSON.stringify(payload),
      })

      return await response.text()
    } catch (error) {
      return `An error occurred while fetching Crunchbase data: ${String(error)}`
    }
  },
  {
    name: 'crunchbase_company_search',
    description:
      'Fetches Crunchbase Basic API data which can be useful to fill in company data. Accepts the name of the company and returns the API result.',
    schema: z.object({
      companyName: z.string().describe('The name of the company to look up'),
    }),
  },
)
