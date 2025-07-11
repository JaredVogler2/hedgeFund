"""
News Sentiment Analysis System with OpenAI Integration
Professional news analysis for trading signals
"""

import openai
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import asyncio
import aiohttp
from collections import defaultdict
import feedparser
import yfinance as yf
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup
import re
import time
from functools import lru_cache
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class NewsConfig:
    """Configuration for news analysis"""
    # API Keys
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    newsapi_key: str = os.getenv('NEWSAPI_KEY', '')
    
    # OpenAI Settings
    openai_model: str = "gpt-4"
    max_tokens: int = 500
    temperature: float = 0.1
    
    # News Sources
    news_sources: List[str] = field(default_factory=lambda: [
        'reuters', 'bloomberg', 'wsj', 'financial-times',
        'cnbc', 'marketwatch', 'seeking-alpha', 'yahoo-finance'
    ])
    
    # Analysis Settings
    lookback_hours: int = 24
    max_articles_per_symbol: int = 20
    sentiment_cache_hours: int = 6
    
    # Sentiment Thresholds
    bullish_threshold: float = 0.6
    bearish_threshold: float = -0.6
    
    # Event Categories
    event_categories: List[str] = field(default_factory=lambda: [
        'earnings', 'guidance', 'merger_acquisition', 'product_launch',
        'regulatory', 'management_change', 'analyst_upgrade', 'analyst_downgrade',
        'partnership', 'legal', 'market_trend', 'economic_indicator'
    ])

@dataclass
class NewsArticle:
    """Individual news article"""
    title: str
    source: str
    published_at: datetime
    url: str
    content: str
    symbols: List[str]
    sentiment_score: Optional[float] = None
    sentiment_reasoning: Optional[str] = None
    event_type: Optional[str] = None
    relevance_score: Optional[float] = None
    
@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    symbol: str
    timestamp: datetime
    overall_sentiment: float  # -1 to 1
    confidence: float  # 0 to 1
    article_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    key_events: List[str]
    sentiment_trend: str  # 'improving', 'deteriorating', 'stable'
    top_articles: List[NewsArticle]

class NewsAnalyzer:
    """Main news analysis system"""
    
    def __init__(self, config: Optional[NewsConfig] = None):
        self.config = config or NewsConfig()
        self._validate_config()
        
        # Initialize OpenAI
        openai.api_key = self.config.openai_api_key
        
        # Initialize news APIs
        self.newsapi = NewsApiClient(api_key=self.config.newsapi_key) if self.config.newsapi_key else None
        
        # Initialize caches
        self.sentiment_cache: Dict[str, SentimentAnalysis] = {}
        self.article_cache: Dict[str, List[NewsArticle]] = {}
        
        # Initialize components
        self.news_fetcher = NewsFetcher(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.event_extractor = EventExtractor(self.config)
        
    def _validate_config(self):
        """Validate configuration"""
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        if not self.config.newsapi_key:
            logger.warning("NewsAPI key not configured, some sources will be unavailable")
    
    def analyze_symbols(self, symbols: List[str]) -> Dict[str, SentimentAnalysis]:
        """Analyze sentiment for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            # Check cache
            if self._is_cache_valid(symbol):
                results[symbol] = self.sentiment_cache[symbol]
                continue
            
            # Fetch and analyze news
            try:
                analysis = self._analyze_symbol(symbol)
                results[symbol] = analysis
                
                # Update cache
                self.sentiment_cache[symbol] = analysis
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results[symbol] = self._create_neutral_analysis(symbol)
        
        return results
    
    def _analyze_symbol(self, symbol: str) -> SentimentAnalysis:
        """Analyze sentiment for a single symbol"""
        logger.info(f"Analyzing news sentiment for {symbol}")
        
        # Fetch news articles
        articles = self.news_fetcher.fetch_news(symbol, self.config.lookback_hours)
        
        if not articles:
            logger.warning(f"No news found for {symbol}")
            return self._create_neutral_analysis(symbol)
        
        # Limit articles
        articles = articles[:self.config.max_articles_per_symbol]
        
        # Analyze each article
        analyzed_articles = []
        for article in articles:
            try:
                # Analyze sentiment
                sentiment, reasoning = self.sentiment_analyzer.analyze_article(article, symbol)
                article.sentiment_score = sentiment
                article.sentiment_reasoning = reasoning
                
                # Extract events
                event_type = self.event_extractor.extract_event_type(article)
                article.event_type = event_type
                
                # Calculate relevance
                relevance = self._calculate_relevance(article, symbol)
                article.relevance_score = relevance
                
                analyzed_articles.append(article)
                
            except Exception as e:
                logger.error(f"Error analyzing article: {e}")
                continue
        
        # Aggregate sentiment
        sentiment_analysis = self._aggregate_sentiment(symbol, analyzed_articles)
        
        return sentiment_analysis
    
    def _calculate_relevance(self, article: NewsArticle, symbol: str) -> float:
        """Calculate article relevance to symbol"""
        relevance = 0.0
        
        # Title mentions
        symbol_mentions_title = article.title.upper().count(symbol.upper())
        relevance += min(symbol_mentions_title * 0.3, 0.6)
        
        # Content mentions
        if article.content:
            symbol_mentions_content = article.content.upper().count(symbol.upper())
            relevance += min(symbol_mentions_content * 0.05, 0.3)
        
        # Recency (more recent = more relevant)
        hours_old = (datetime.now() - article.published_at).total_seconds() / 3600
        recency_score = max(0, 1 - (hours_old / self.config.lookback_hours))
        relevance += recency_score * 0.1
        
        return min(relevance, 1.0)
    
    def _aggregate_sentiment(self, symbol: str, articles: List[NewsArticle]) -> SentimentAnalysis:
        """Aggregate sentiment from multiple articles"""
        if not articles:
            return self._create_neutral_analysis(symbol)
        
        # Weight by relevance and recency
        weighted_sentiments = []
        weights = []
        
        for article in articles:
            if article.sentiment_score is not None:
                # Calculate weight
                recency_weight = self._calculate_recency_weight(article.published_at)
                relevance_weight = article.relevance_score or 0.5
                weight = recency_weight * relevance_weight
                
                weighted_sentiments.append(article.sentiment_score * weight)
                weights.append(weight)
        
        # Calculate overall sentiment
        if weights:
            overall_sentiment = sum(weighted_sentiments) / sum(weights)
        else:
            overall_sentiment = 0.0
        
        # Count sentiment categories
        bullish_count = sum(1 for a in articles if a.sentiment_score and a.sentiment_score > self.config.bullish_threshold)
        bearish_count = sum(1 for a in articles if a.sentiment_score and a.sentiment_score < self.config.bearish_threshold)
        neutral_count = len(articles) - bullish_count - bearish_count
        
        # Extract key events
        key_events = []
        for article in articles:
            if article.event_type and article.event_type != 'market_trend':
                key_events.append(f"{article.event_type}: {article.title[:50]}")
        
        # Determine sentiment trend
        sentiment_trend = self._determine_sentiment_trend(articles)
        
        # Get top articles
        top_articles = sorted(
            articles,
            key=lambda a: (a.relevance_score or 0) * abs(a.sentiment_score or 0),
            reverse=True
        )[:5]
        
        # Calculate confidence
        confidence = self._calculate_confidence(articles, overall_sentiment)
        
        return SentimentAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            article_count=len(articles),
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            key_events=key_events[:5],
            sentiment_trend=sentiment_trend,
            top_articles=top_articles
        )
    
    def _calculate_recency_weight(self, published_at: datetime) -> float:
        """Calculate weight based on article recency"""
        hours_old = (datetime.now() - published_at).total_seconds() / 3600
        
        # Exponential decay
        decay_rate = 0.1
        weight = np.exp(-decay_rate * hours_old)
        
        return weight
    
    def _determine_sentiment_trend(self, articles: List[NewsArticle]) -> str:
        """Determine if sentiment is improving or deteriorating"""
        if len(articles) < 3:
            return 'stable'
        
        # Sort by time
        sorted_articles = sorted(articles, key=lambda a: a.published_at)
        
        # Compare recent vs older sentiment
        midpoint = len(sorted_articles) // 2
        
        older_sentiment = np.mean([a.sentiment_score for a in sorted_articles[:midpoint] 
                                 if a.sentiment_score is not None])
        recent_sentiment = np.mean([a.sentiment_score for a in sorted_articles[midpoint:] 
                                  if a.sentiment_score is not None])
        
        if recent_sentiment > older_sentiment + 0.1:
            return 'improving'
        elif recent_sentiment < older_sentiment - 0.1:
            return 'deteriorating'
        else:
            return 'stable'
    
    def _calculate_confidence(self, articles: List[NewsArticle], overall_sentiment: float) -> float:
        """Calculate confidence in sentiment analysis"""
        if not articles:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Number of articles
        article_factor = min(len(articles) / 10, 1.0)  # Max confidence at 10+ articles
        
        # 2. Agreement between articles
        sentiments = [a.sentiment_score for a in articles if a.sentiment_score is not None]
        if sentiments:
            sentiment_std = np.std(sentiments)
            agreement_factor = max(0, 1 - sentiment_std)
        else:
            agreement_factor = 0.0
        
        # 3. Strength of sentiment
        strength_factor = min(abs(overall_sentiment), 1.0)
        
        # 4. Quality of sources
        quality_sources = ['reuters', 'bloomberg', 'wsj', 'financial-times']
        quality_ratio = sum(1 for a in articles if any(q in a.source.lower() for q in quality_sources)) / len(articles)
        
        # Combine factors
        confidence = (article_factor * 0.3 + 
                     agreement_factor * 0.3 + 
                     strength_factor * 0.2 + 
                     quality_ratio * 0.2)
        
        return confidence
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached sentiment is still valid"""
        if symbol not in self.sentiment_cache:
            return False
        
        cached = self.sentiment_cache[symbol]
        age_hours = (datetime.now() - cached.timestamp).total_seconds() / 3600
        
        return age_hours < self.config.sentiment_cache_hours
    
    def _create_neutral_analysis(self, symbol: str) -> SentimentAnalysis:
        """Create neutral sentiment when no news available"""
        return SentimentAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_sentiment=0.0,
            confidence=0.0,
            article_count=0,
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
            key_events=[],
            sentiment_trend='stable',
            top_articles=[]
        )
    
    def get_market_sentiment(self) -> Dict[str, float]:
        """Get overall market sentiment"""
        # Analyze major indices
        indices = ['SPY', 'QQQ', 'DIA', 'IWM']
        
        sentiments = {}
        for index in indices:
            analysis = self._analyze_symbol(index)
            sentiments[index] = analysis.overall_sentiment
        
        # Overall market sentiment
        market_sentiment = np.mean(list(sentiments.values()))
        sentiments['MARKET'] = market_sentiment
        
        return sentiments

class NewsFetcher:
    """Fetches news from multiple sources"""
    
    def __init__(self, config: NewsConfig):
        self.config = config
        self.session = None
        
    def fetch_news(self, symbol: str, lookback_hours: int) -> List[NewsArticle]:
        """Fetch news from all configured sources"""
        all_articles = []
        
        # Fetch from different sources
        if self.config.newsapi_key:
            all_articles.extend(self._fetch_newsapi(symbol, lookback_hours))
        
        all_articles.extend(self._fetch_yahoo_finance(symbol, lookback_hours))
        all_articles.extend(self._fetch_rss_feeds(symbol, lookback_hours))
        
        # Remove duplicates
        unique_articles = self._remove_duplicates(all_articles)
        
        # Sort by date
        unique_articles.sort(key=lambda a: a.published_at, reverse=True)
        
        return unique_articles
    
    def _fetch_newsapi(self, symbol: str, lookback_hours: int) -> List[NewsArticle]:
        """Fetch from NewsAPI"""
        articles = []
        
        try:
            from_date = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()
            
            # Get company name for better search
            ticker = yf.Ticker(symbol)
            company_name = ticker.info.get('longName', symbol)
            
            # Search for articles
            newsapi = NewsApiClient(api_key=self.config.newsapi_key)
            response = newsapi.get_everything(
                q=f'"{company_name}" OR "{symbol}"',
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=20
            )
            
            for article_data in response.get('articles', []):
                article = NewsArticle(
                    title=article_data['title'],
                    source=article_data['source']['name'],
                    published_at=datetime.fromisoformat(article_data['publishedAt'].replace('Z', '+00:00')),
                    url=article_data['url'],
                    content=article_data.get('description', ''),
                    symbols=[symbol]
                )
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
        
        return articles
    
    def _fetch_yahoo_finance(self, symbol: str, lookback_hours: int) -> List[NewsArticle]:
        """Fetch from Yahoo Finance"""
        articles = []
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            for item in news:
                # Parse publish time
                pub_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                
                if pub_time < cutoff_time:
                    continue
                
                article = NewsArticle(
                    title=item.get('title', ''),
                    source='Yahoo Finance',
                    published_at=pub_time,
                    url=item.get('link', ''),
                    content=item.get('summary', ''),
                    symbols=[symbol]
                )
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance: {e}")
        
        return articles
    
    def _fetch_rss_feeds(self, symbol: str, lookback_hours: int) -> List[NewsArticle]:
        """Fetch from RSS feeds"""
        articles = []
        
        # RSS feeds for financial news
        rss_feeds = {
            'MarketWatch': f'https://feeds.marketwatch.com/marketwatch/topstories/',
            'Seeking Alpha': f'https://seekingalpha.com/symbol/{symbol}/feed',
            'Reuters': 'https://www.reutersagency.com/feed/',
        }
        
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        for source, feed_url in rss_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:  # Limit entries
                    # Check if symbol mentioned
                    if symbol.upper() not in entry.title.upper() and symbol.upper() not in entry.get('summary', '').upper():
                        continue
                    
                    # Parse date
                    pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
                    
                    if pub_date < cutoff_time:
                        continue
                    
                    article = NewsArticle(
                        title=entry.title,
                        source=source,
                        published_at=pub_date,
                        url=entry.link,
                        content=entry.get('summary', ''),
                        symbols=[symbol]
                    )
                    articles.append(article)
                    
            except Exception as e:
                logger.error(f"Error fetching from {source}: {e}")
        
        return articles
    
    def _remove_duplicates(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple duplicate check based on title
            title_key = article.title.lower().strip()
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles

class SentimentAnalyzer:
    """Analyzes sentiment using OpenAI"""
    
    def __init__(self, config: NewsConfig):
        self.config = config
        
    def analyze_article(self, article: NewsArticle, symbol: str) -> Tuple[float, str]:
        """Analyze sentiment of a single article"""
        
        # Create prompt
        prompt = self._create_sentiment_prompt(article, symbol)
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at interpreting news sentiment for stock trading. Provide precise sentiment scores."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Parse response
            content = response.choices[0].message.content
            sentiment_score, reasoning = self._parse_sentiment_response(content)
            
            return sentiment_score, reasoning
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0, "Error in analysis"
    
    def _create_sentiment_prompt(self, article: NewsArticle, symbol: str) -> str:
        """Create prompt for sentiment analysis"""
        
        prompt = f"""Analyze the sentiment of this news article for {symbol} stock:

Title: {article.title}
Source: {article.source}
Date: {article.published_at}

Content: {article.content[:500]}...

Please provide:
1. A sentiment score from -1.0 (very bearish) to 1.0 (very bullish)
2. Brief reasoning for your score (max 100 words)

Consider:
- Impact on stock price
- Time horizon of impact
- Credibility of information
- Market reaction likelihood

Format your response as:
SENTIMENT_SCORE: [score]
REASONING: [your reasoning]
"""
        
        return prompt
    
    def _parse_sentiment_response(self, response: str) -> Tuple[float, str]:
        """Parse sentiment analysis response"""
        try:
            lines = response.strip().split('\n')
            
            sentiment_score = 0.0
            reasoning = ""
            
            for line in lines:
                if line.startswith('SENTIMENT_SCORE:'):
                    score_str = line.replace('SENTIMENT_SCORE:', '').strip()
                    sentiment_score = float(score_str)
                    sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1, 1]
                    
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
            
            return sentiment_score, reasoning
            
        except Exception as e:
            logger.error(f"Error parsing sentiment response: {e}")
            return 0.0, "Parse error"

class EventExtractor:
    """Extracts event types from news"""
    
    def __init__(self, config: NewsConfig):
        self.config = config
        
        # Keywords for event detection
        self.event_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'eps', 'beat', 'miss', 'quarterly results'],
            'guidance': ['guidance', 'forecast', 'outlook', 'raises', 'lowers', 'maintains'],
            'merger_acquisition': ['merger', 'acquisition', 'acquire', 'deal', 'buyout', 'takeover'],
            'product_launch': ['launch', 'unveil', 'introduce', 'release', 'announce new'],
            'regulatory': ['fda', 'sec', 'regulatory', 'approval', 'investigation', 'compliance'],
            'management_change': ['ceo', 'cfo', 'resign', 'appoint', 'hire', 'departure'],
            'analyst_upgrade': ['upgrade', 'outperform', 'buy rating', 'price target raised'],
            'analyst_downgrade': ['downgrade', 'underperform', 'sell rating', 'price target cut'],
            'partnership': ['partnership', 'collaboration', 'joint venture', 'strategic alliance'],
            'legal': ['lawsuit', 'legal', 'court', 'settlement', 'litigation'],
        }
    
    def extract_event_type(self, article: NewsArticle) -> Optional[str]:
        """Extract primary event type from article"""
        
        # Combine title and content for analysis
        text = (article.title + ' ' + (article.content or '')).lower()
        
        # Score each event type
        event_scores = {}
        
        for event_type, keywords in self.event_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                event_scores[event_type] = score
        
        # Return highest scoring event type
        if event_scores:
            return max(event_scores.items(), key=lambda x: x[1])[0]
        
        return 'market_trend'  # Default category

class SentimentAggregator:
    """Aggregates sentiment across multiple symbols"""
    
    def __init__(self):
        self.sector_mappings = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Finance': ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'CVS', 'ABBV'],
            'Consumer': ['AMZN', 'WMT', 'HD', 'NKE', 'MCD'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG']
        }
    
    def calculate_sector_sentiment(self, symbol_sentiments: Dict[str, SentimentAnalysis]) -> Dict[str, float]:
        """Calculate sentiment by sector"""
        sector_sentiments = {}
        
        for sector, symbols in self.sector_mappings.items():
            sector_scores = []
            
            for symbol in symbols:
                if symbol in symbol_sentiments:
                    analysis = symbol_sentiments[symbol]
                    if analysis.confidence > 0.3:  # Only include confident analyses
                        sector_scores.append(analysis.overall_sentiment)
            
            if sector_scores:
                sector_sentiments[sector] = np.mean(sector_scores)
            else:
                sector_sentiments[sector] = 0.0
        
        return sector_sentiments


# Example usage
if __name__ == "__main__":
    # Initialize configuration
    config = NewsConfig()
    
    # Check if API keys are set
    if not config.openai_api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Initialize analyzer
    analyzer = NewsAnalyzer(config)
    
    # Analyze some symbols
    symbols = ['AAPL', 'MSFT', 'TSLA']
    
    print("Analyzing news sentiment...")
    results = analyzer.analyze_symbols(symbols)
    
    # Display results
    for symbol, analysis in results.items():
        print(f"\n{symbol} Sentiment Analysis:")
        print(f"  Overall Sentiment: {analysis.overall_sentiment:.2f} ({analysis.confidence:.0%} confidence)")
        print(f"  Articles Analyzed: {analysis.article_count}")
        print(f"  Sentiment Breakdown: {analysis.bullish_count} bullish, "
              f"{analysis.bearish_count} bearish, {analysis.neutral_count} neutral")
        print(f"  Trend: {analysis.sentiment_trend}")
        
        if analysis.key_events:
            print(f"  Key Events:")
            for event in analysis.key_events:
                print(f"    - {event}")
        
        if analysis.top_articles:
            print(f"  Top Articles:")
            for article in analysis.top_articles[:3]:
                print(f"    - {article.title[:80]}...")
                print(f"      Sentiment: {article.sentiment_score:.2f}, "
                      f"Relevance: {article.relevance_score:.2f}")
    
    # Get market sentiment
    print("\nMarket Sentiment:")
    market_sentiment = analyzer.get_market_sentiment()
    for index, sentiment in market_sentiment.items():
        print(f"  {index}: {sentiment:.2f}")
    
    # Calculate sector sentiment
    aggregator = SentimentAggregator()
    sector_sentiments = aggregator.calculate_sector_sentiment(results)
    
    print("\nSector Sentiments:")
    for sector, sentiment in sector_sentiments.items():
        print(f"  {sector}: {sentiment:.2f}")
