#!/usr/bin/env python3
"""
Carta Capital News Scraper
===========================

Scrapes news articles from Carta Capital using binary search to find max pages.
Saves to ../scraped/cartacapital/ directory.

Usage:
    python3 scrape_cartacapital.py [--max-pages N] [--start-page N] [--no-auto-detect]
"""

import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import argparse
from datetime import datetime

# Configuration
BASE_URL = "https://www.cartacapital.com.br/mais-recentes/page/{}/"
PAGE_1_URL = "https://www.cartacapital.com.br/mais-recentes/"
OUTPUT_DIR = Path(__file__).parent.parent / "scraped" / "cartacapital"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Rate limiting
DELAY_BETWEEN_ARTICLES = 0.5  # seconds
DELAY_BETWEEN_PAGES = 1.0     # seconds


def scrape_article(url: str, session: requests.Session) -> dict:
    """
    Scrape a single article from Carta Capital
    
    Args:
        url: Article URL
        session: Requests session
        
    Returns:
        Dictionary with article data or None if failed
    """
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ''
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])
        
        # Extract date from meta tag
        date = ''
        meta_date = soup.find('meta', property='article:published_time')
        if meta_date:
            date = meta_date.get('content', '')
        
        if not title or not text:
            return None
        
        return {
            'url': url,
            'title': title,
            'text': text,
            'date': date,
            'portal': 'cartacapital',
            'scraped_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"  ✗ Error scraping {url}: {e}")
        return None


def scrape_page(page_num: int, session: requests.Session) -> list:
    """
    Scrape all article links from a listing page
    
    Args:
        page_num: Page number
        session: Requests session
        
    Returns:
        List of article URLs
    """
    # Page 1 uses /mais-recentes/ (no page number), pages 2+ use /page/2/, /page/3/, etc.
    if page_num == 1:
        url = PAGE_1_URL
    else:
        url = BASE_URL.format(page_num)
    
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find article links
        links = soup.find_all('a', href=True)
        article_links = []
        
        for a in links:
            href = a['href']
            
            # Filter for actual article URLs
            if ('cartacapital.com.br' in href 
                and '/' in href.split('cartacapital.com.br/')[-1]
                and '/cupons/' not in href
                and '/mais-recentes/' not in href
                and '/page/' not in href
                and '/studiocarta/' not in href
                and '/blogs/' not in href
                and '/wp-content/' not in href
                and '/cartaexpressa/' not in href
                and '/edicao/' not in href
                and href not in ['https://www.cartacapital.com.br/', 'https://www.cartacapital.com.br']):
                article_links.append(href)
        
        # Remove duplicates
        article_links = list(set(article_links))
        
        return article_links
        
    except Exception as e:
        print(f"  ✗ Error on page {page_num}: {e}")
        return []


def find_max_page(session: requests.Session) -> int:
    """
    Use binary search to find the maximum page number
    
    Args:
        session: Requests session
        
    Returns:
        Maximum page number
    """
    print("\n" + "="*80)
    print("BUSCA AUTOMÁTICA DO NÚMERO MÁXIMO DE PÁGINAS")
    print("="*80 + "\n")
    
    # Phase 1: Exponential search to find upper bound
    print("Fase 1: Busca exponencial para encontrar limite superior...")
    test_page = 1
    while test_page <= 10000:  # Safety limit
        article_links = scrape_page(test_page, session)
        
        if not article_links or len(article_links) == 0:
            print(f"  Testando página {test_page}... ✗ Vazia")
            break
        
        print(f"  Testando página {test_page}... ✓ {len(article_links)} artigos encontrados")
        test_page *= 2
        time.sleep(0.5)
    
    # Phase 2: Binary search between last valid and first invalid
    left = test_page // 2
    right = test_page
    max_page = left
    
    print(f"\nFase 2: Busca binária entre páginas {left} e {right}...")
    
    while left <= right:
        mid = (left + right) // 2
        article_links = scrape_page(mid, session)
        
        if article_links and len(article_links) > 0:
            print(f"  Testando página {mid}... ✓ {len(article_links)} artigos")
            max_page = mid
            left = mid + 1
        else:
            print(f"  Testando página {mid}... ✗ Vazia")
            right = mid - 1
        
        time.sleep(0.5)
    
    print(f"\n✓ Número máximo de páginas encontrado: {max_page}")
    print("="*80 + "\n")
    
    return max_page


def get_existing_urls() -> set:
    """
    Get URLs of already scraped articles
    
    Returns:
        Set of URLs
    """
    existing_urls = set()
    
    if OUTPUT_DIR.exists():
        for json_file in OUTPUT_DIR.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                    existing_urls.add(article['url'])
            except:
                pass
    
    return existing_urls


def save_article(article: dict, index: int):
    """
    Save article to JSON file
    
    Args:
        article: Article dictionary
        index: File index number
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT_DIR / f"cartacapital_{index:04d}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(article, f, ensure_ascii=False, indent=2)


def main(max_pages: int = None, start_page: int = 1, auto_detect: bool = True):
    """
    Main scraping function
    
    Args:
        max_pages: Maximum number of pages to scrape (None = auto-detect)
        start_page: Starting page number
        auto_detect: If True, automatically detect max pages using binary search
    """
    print("\n" + "="*80)
    print("CARTA CAPITAL NEWS SCRAPER")
    print("="*80 + "\n")
    
    # Setup
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    
    # Get existing articles
    existing_urls = get_existing_urls()
    print(f"Already scraped: {len(existing_urls)} articles\n")
    
    # Auto-detect max pages if requested
    if auto_detect and max_pages is None:
        max_pages = find_max_page(session)
    elif max_pages is None:
        max_pages = 100  # Default fallback
    
    # Start scraping
    total_articles = len(existing_urls)
    new_articles = 0
    skipped = 0
    
    print(f"Starting from page {start_page}, max {max_pages} pages\n")
    
    for page_num in range(start_page, max_pages + 1):
        print(f"Page {page_num}:")
        
        # Get article links from page
        article_urls = scrape_page(page_num, session)
        
        if not article_urls:
            print(f"  No articles found - stopping")
            break
        
        print(f"  Found {len(article_urls)} article links")
        
        # Scrape each article
        for url in article_urls:
            # Skip if already scraped
            if url in existing_urls:
                skipped += 1
                continue
            
            # Scrape article
            article = scrape_article(url, session)
            
            if article:
                total_articles += 1
                new_articles += 1
                save_article(article, total_articles)
                print(f"  ✓ [{total_articles}] {article['title'][:60]}...")
                existing_urls.add(url)
            
            time.sleep(DELAY_BETWEEN_ARTICLES)
        
        print(f"  Progress: {new_articles} new, {skipped} skipped\n")
        time.sleep(DELAY_BETWEEN_PAGES)
    
    # Summary
    print("\n" + "="*80)
    print("SCRAPING COMPLETED")
    print("="*80)
    print(f"Total articles: {total_articles}")
    print(f"New articles: {new_articles}")
    print(f"Already existed: {len(existing_urls) - new_articles}")
    print(f"Saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape Carta Capital news articles')
    parser.add_argument('--max-pages', type=int, help='Maximum number of pages to scrape')
    parser.add_argument('--start-page', type=int, default=1, help='Starting page number')
    parser.add_argument('--no-auto-detect', action='store_true', help='Disable automatic page detection')
    
    args = parser.parse_args()
    
    main(
        max_pages=args.max_pages,
        start_page=args.start_page,
        auto_detect=not args.no_auto_detect
    )
