#Import required libraries:
import csv
import time
from unidecode import unidecode
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException

def data_collection(keyword, max_pages=2, filename='data/raw_data/product_reviews.csv'):
    """
    A function that ties all the smaller functions together to scrape Amazon product reviews.

    Parameters:
    - keyword (str): The search term to look for on Amazon.
    - max_pages (int): The maximum number of search result pages to scrape.
    - filename (str): Name of the CSV file to which the reviews will be exported.

    """
    # Step 1: Initialize the Selenium driver and navigate to Amazon's website
    driver = setup_driver()

    # Step 2: Perform a search on Amazon
    amazon_search(driver, keyword)

    # Step 3: Collect product links
    links = collect_product_links(driver, max_pages)

    # Step 4: Scrape product reviews
    product_reviews = scrape_reviews(driver, links)

    # Step 5: Export reviews to CSV
    export_reviews_to_csv(product_reviews, filename)


def setup_driver(url='https://www.amazon.com', driver_path='D:/Programming/chromedriver_win32/chromedriver', headless=False):
    """
    Sets up a Chrome driver for Selenium and navigates to the given URL.
    
    Parameters:
    - url: The web page to navigate to.
    - driver_path: Path to the chromedriver executable.
    - headless: Boolean indicating whether to run Chrome in headless mode.
    
    Returns:
    - driver: The Selenium Chrome WebDriver object.
    """
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('--headless')

    driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))
    wait = WebDriverWait(driver, 20)

    driver.get(url)
    
    return driver


def amazon_search(driver, keyword):
    """
    Searches for a given keyword on Amazon using a Selenium driver.
    
    Parameters:
    - driver: The Selenium WebDriver object.
    - keyword: The search term to look for on Amazon.
    """
    driver.implicitly_wait(20)

    search = driver.find_element(By.ID, 'twotabsearchtextbox')
    search.send_keys(keyword)

    search_button = driver.find_element(By.ID, 'nav-search-submit-button')
    search_button.click()

    driver.implicitly_wait(20)


def collect_product_links(driver, max_pages=2):
    """
    Collects product links from Amazon search results.

    Parameters:
    - driver: The Selenium WebDriver object.
    - max_pages: The maximum number of search result pages to scrape. Default is 20.

    Returns:
    - product_link: List containing product links from the search results.
    """
    product_link = []
    current_page = 1

    # Loop through all pages of search results
    while True:
        time.sleep(5)  # Waiting for the page to load
        try:
            items = driver.find_elements(By.XPATH, '//div[contains(@class, "s-result-item s-asin")]')
            for item in items:
                link = item.find_element(By.XPATH, './/a[@class="a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal"]').get_attribute("href")
                product_link.append(link)

            if current_page < max_pages:
                try:
                    next_search_page_button = driver.find_element(By.CLASS_NAME, 's-pagination-next')
                    next_search_page_button.click()
                    time.sleep(5)  # Give it some time to load
                    current_page += 1
                except NoSuchElementException:
                    # Handle when the "Next" button is not found, possibly indicating the last page
                    print("No more pages found.")
                    break
            else:
                print("Reached the maximum number of pages to scrape.")
                break

        except Exception as e:
            print(f"Error while extracting product links or navigating: {e}")
            break

    return product_link


def scrape_reviews(driver, links):
    
    # Setup WebDriverWait
    wait = WebDriverWait(driver, 10)
    
    product_reviews = []

    for link in links[:2]:
        driver.get(link)
        try:
            # Click on the "See all reviews" link
            all_reviews_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a')))
            all_reviews_button.click()

            # After clicking on "See all reviews" link
            page_number = 1
            while True:
                try:
                    # Collect reviews as before
                    review_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[class='a-section celwidget']")))
                    for review_element in review_elements[:2]:
                        try:
                            review_text = review_element.find_element(By.CSS_SELECTOR, '[data-hook="review-body"]').text
                        except:
                            review_text = " "
                        try:
                            review_rating = review_element.find_element(By.CSS_SELECTOR, '[class="a-icon-alt"]').get_attribute('innerHTML')
                        except:
                            review_rating = " "
                        try:
                            profile_name = review_element.find_element(By.CSS_SELECTOR, '[class="a-profile-name"]').text
                        except:
                            profile_name = " "
                        try:
                            review_title = review_element.find_element(By.XPATH, './/div[2]/a/span[2]').text
                        except:
                            review_title = " "

                        product_reviews.append([link, profile_name, review_rating, review_title, review_text])

                    # Try to find and click the "Next" button to go to the next page of reviews
                    next_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="cm_cr-pagination_bar"]/ul/li[2]/a')))
                    if not next_button:
                        break  # Exit if no "Next" button found

                    next_button.click()
                    time.sleep(1)  # Give it some time to load
                    page_number += 1
                except TimeoutException:
                    # Handle when the "Next" button is not found, possibly indicating the last page
                    break
                except:
                    # Handle other exceptions, e.g., log the error and continue
                    pass
        except Exception as e:
                    print(f"Error while processing {link}: {e}")
    
    # Close the browser after scraping
    driver.quit()

    return product_reviews
    

def clean_text(text):
    """
    Clean the provided text by removing any non-ASCII characters using unidecode.
    
    Parameters:
    - text (str or None): The text to be cleaned. If None, an empty string will be returned.
    
    Returns:
    - str: The cleaned text with non-ASCII characters removed. If the input is None, an empty string is returned.
    """
    if text is None:
        return ""
    return unidecode(text)

def export_reviews_to_csv(reviews, filename = 'data/raw_data/product_reviews.csv'):
    """
    Export a list of reviews to a CSV file.

    Parameters:
    - reviews: A list of reviews. Each review should be a list in the format [Link, Name, Rating, Review_title, Review].
    - filename: Name of the CSV file to which the reviews will be exported. Default is 'product_reviews.csv'.
    
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Link', 'Name', 'Rating', 'Review_title', 'Review'])  # Writing the header
        
        for review in reviews:
            link, profile_name, rating, review_title, review_text = review
            cleaned_profile_name = clean_text(profile_name)
            cleaned_review_title = clean_text(review_title)
            cleaned_review_text = clean_text(review_text)
            
            writer.writerow([link, cleaned_profile_name, rating, cleaned_review_title, cleaned_review_text])