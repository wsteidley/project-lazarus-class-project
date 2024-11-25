import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime, timezone

# Scraped page which can be used for testing
# with open("articles_page.txt", "r") as file:
#     fake_content = file.read()
# # print(fake_content)

# class FakeResponse:
#     def __init__(self, status_code, content):
#         self.status_code = status_code
#         self.content = content

# use this timestamp for articles index and output file so they can be easily related
utc_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
print(utc_timestamp)

def scrape_techcrunch_listings(url):
    response = requests.get(url)

    # Used for troubleshooting but is not 1:1 at the moment so code needs work to make test work better
    # response = FakeResponse(200, fake_content)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.select('ul.wp-block-post-template > li.wp-block-post')
        data = []

        for article in articles:
            try:
                title_element = article.select_one('h3.loop-card__title')
                title = title_element.text.strip() if title_element else ''
                url = title_element.find('a')['href'] if title_element else ''
                author = article.find('a', class_='loop-card__author').get_text()
                publication_date = article.select_one('time')['datetime'] if article.select_one('time') else ''
                # summary = article.select_one('p.wp-block-post-excerpt__excerpt').text.strip() if article.select_one('p.wp-block-post-excerpt__excerpt') else ''
            except Exception as e:
                print(f"Exception for url: {url}. Exception: {e}")
            else:
                data.append({
                    'title': title,
                    'url': url,
                    'author': author,
                    'publication_date': publication_date,
                    'timestamp': utc_timestamp
                    # 'summary': summary
                })


        return data
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None

# This should work but we don't need to paginate until we're ready to start
# scraping all the data
def scrape_techcrunch_with_pagination(base_url, search_term):
    all_data = []
    page = 0
    search_string = f"?s={search_term}" if search_term else ""
    
    # Only Scrape 1 page. If we want to scrape all pages
    # Use the while True loop
    while page < 1:
    # Scrape data until we get no data back aka 404
    # while True:
        url = f"{base_url}/page/{page}/{search_string}"
        page += 1
        print(f"Scraping page {page} from url: {url}")

        page_data = scrape_techcrunch_listings(url)
        if page_data:
            all_data.extend(page_data)
        else:
            print(f"Failed to retrieve data from page: {page}")
            break
        
    print(f"Length of all data: {len(all_data)}")
    return all_data

def scrape_techcrunch_article(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

       # Extracting the title
        title = soup.select_one('h1.wp-block-post-title').text.strip()

        # Extracting the author
        author_block = soup.select_one('div.post-authors-list__authors') if soup.select_one('div.post-authors-list__authors') else soup.select_one('div.article-hero__authors') 
        author = author_block.find('a').get_text()

        # Extracting the publication date
        publication_date = soup.select_one('div.wp-block-post-date > time')['datetime']

        # Extracting the content
        content = soup.select_one('div.wp-block-post-content').text.strip()

        return {
            'title': title,
            'author': author,
            'publication_date': publication_date,
            'content': content,
            'timestamp': utc_timestamp,
            'url': url
        }
    else:
        print(f"Failed to retrieve the article. Status code: {response.status_code}")
        return None


def save_article_data_to_csv(data, filename):
    if not filename:
        raise Exception('Need to provide a filename to save')
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Article data successfully saved to {filename}")

def scrape_all_article_data(article_urls):
    total_urls = len(article_urls)
    all_article_data = []
    for index, url in enumerate(article_urls):
        try:
            print(f"{index}/{total_urls}: {url}")
            article_data = scrape_techcrunch_article(url)
        except Exception as e:
            print(f"Exception for url: {url}. Exception: {e}")
        else:
            if article_data:
                all_article_data.append(article_data)

    return all_article_data


def generate_article_index_data(url, search_term=""):
    # Get the current UTC time
    if not url:
        raise ValueError('Configure URL if you really want to scrape stuff. Otherwise use articles_page.txt for testing.')
    article_data = scrape_techcrunch_with_pagination(url, search_term)

    # print(json.dumps(article_data, indent=2))
    
    return article_data

if __name__ == "__main__":
    # uses the startup category as a starting point
    # url = 'https://techcrunch.com/category/startups/'
    # only need this to generate the CSV
    # url = 'https://techcrunch.com/tag/climate'

    # Should be a url or read in data from articles_page.txt
    url = ''
    article_index_data = generate_article_index_data(url)
    article_index_filename = f"techcrunch_article_{utc_timestamp}_data_index.csv"
    save_article_data_to_csv(article_index_data, article_index_filename)

    # load csv we just created
    article_index_df = pd.DataFrame(article_index_data)

    article_urls = article_index_df["url"].tolist()
    article_data_scraped = scrape_all_article_data(article_urls)
    techcrunch_article_data_filename = f"techcrunch_article_{utc_timestamp}_data.csv"
    save_article_data_to_csv(article_data_scraped, techcrunch_article_data_filename)

    
