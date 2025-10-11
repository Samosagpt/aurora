"""
Web Services Module for Aurora
Provides Wikipedia, Google, YouTube, Weather, News, and AI-powered web search functionality
"""
import webbrowser
import requests
import logging
import wikipedia
from typing import Optional, List, Dict, Any
from Generation import ollama_manager

try:
    from config_prod import config
except ImportError:
    from config import config

logger = logging.getLogger(__name__)


class WebServices:
    """Web services integration with AI-powered search capabilities"""
    
    def __init__(self):
        self.weather_api_key = config.OPENWEATHER_API_KEY
        self.news_api_key = config.NEWS_API_KEY
        self.ollama_host = config.OLLAMA_HOST
        
    # ============ Wikipedia Integration ============
    
    def search_wikipedia(self, topic: str, sentences: int = 3) -> Optional[str]:
        """
        Search Wikipedia and return summary
        
        Args:
            topic: Topic to search for
            sentences: Number of sentences in summary
            
        Returns:
            Wikipedia summary or None if not found
        """
        try:
            logger.info(f"Searching Wikipedia for: {topic}")
            # Set language to English
            wikipedia.set_lang("en")
            
            # Get summary
            summary = wikipedia.summary(topic, sentences=sentences)
            return summary
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Multiple options found
            logger.warning(f"Disambiguation error for '{topic}': {e.options[:5]}")
            options = e.options[:5]
            return f"Multiple results found. Did you mean:\n" + "\n".join(f"- {opt}" for opt in options)
            
        except wikipedia.exceptions.PageError:
            logger.warning(f"Wikipedia page not found for: {topic}")
            return None
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return None
    
    def search_wikipedia_detailed(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed Wikipedia information
        
        Args:
            topic: Topic to search for
            
        Returns:
            Dict with title, summary, url, and content
        """
        try:
            page = wikipedia.page(topic)
            return {
                "title": page.title,
                "summary": wikipedia.summary(topic, sentences=5),
                "url": page.url,
                "content": page.content[:2000]  # First 2000 chars
            }
        except Exception as e:
            logger.error(f"Detailed Wikipedia search error: {e}")
            return None
    
    # ============ Google Search ============
    
    def open_google(self, query: str) -> str:
        """
        Open Google search in browser
        
        Args:
            query: Search query
            
        Returns:
            Confirmation message
        """
        try:
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(url)
            logger.info(f"Opened Google search for: {query}")
            return f"🔍 Opening Google search for '{query}'..."
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return f"❌ Failed to open Google search: {e}"
    
    def ai_web_search(self, query: str, model: str = None) -> str:
        """
        Perform AI-powered web search using Ollama's official Web Search API
        
        Args:
            query: Search query
            model: Ollama model to use for summarization (optional)
            
        Returns:
            Formatted search results with AI summary
        """
        try:
            logger.info(f"AI web search for: {query}")
            
            # Use Ollama's official web search API
            import ollama
            
            try:
                # Initialize Ollama client with authentication
                client_config = {'host': self.ollama_host}
                
                # Add API key if provided (required for web search)
                if hasattr(config, 'OLLAMA_API_KEY') and config.OLLAMA_API_KEY:
                    client_config['headers'] = {
                        'Authorization': f'Bearer {config.OLLAMA_API_KEY}'
                    }
                    logger.info("Using API key for web search authentication")
                else:
                    logger.warning("No OLLAMA_API_KEY found - web search may fail")
                
                # Create authenticated client
                client = ollama.Client(**client_config)
                
                # Call Ollama's web_search function through the client
                search_response = client.web_search(query, max_results=5)
                
                # Check if we have results (response is WebSearchResponse object)
                if not search_response or not hasattr(search_response, 'results') or not search_response.results:
                    logger.warning(f"No search results found for: {query}")
                    return "❌ No results found for your query."
                
                # Format the search results
                formatted_results = []
                for i, result in enumerate(search_response.results[:5], 1):
                    title = result.title if hasattr(result, 'title') else 'Untitled'
                    url = result.url if hasattr(result, 'url') else ''
                    content = result.content[:500] if hasattr(result, 'content') else ''  # Show more content
                    
                    formatted_results.append(
                        f"**{i}. {title}**\n"
                        f"   🔗 {url}\n"
                        f"   � {content}\n"
                    )
                
                sources_text = "\n".join(formatted_results)
                
                # Return the formatted search results directly
                return f"🔍 **Web Search Results: '{query}'**\n\n{sources_text}\n\n💡 *Data from real-time web search*"
                    
            except AttributeError:
                # Fallback if ollama.web_search not available
                logger.warning("Ollama web_search API not available, using chat fallback")
                
                search_prompt = f"""You are a web search assistant. Provide comprehensive information about: "{query}"

Include:
1. A brief overview/definition
2. Key facts and details  
3. Current relevance or recent developments
4. Related topics to explore

Format clearly with headers and bullets."""

                response = ollama_manager.chat_with_memory(search_prompt, model_name=model)
                return f"🔍 **AI Search Results for '{query}':**\n\n{response}"
            
        except Exception as e:
            logger.error(f"AI web search error: {e}")
            return f"❌ Web search failed: {str(e)}\n\nPlease ensure:\n1. OLLAMA_API_KEY is set in .env\n2. Ollama Python library is up to date: pip install --upgrade ollama"
    
    def smart_search(self, query: str) -> str:
        """
        Intelligent search that combines Wikipedia and AI knowledge
        
        Args:
            query: Search query
            
        Returns:
            Combined search results
        """
        try:
            results = []
            
            # Try Wikipedia first
            wiki_result = self.search_wikipedia(query, sentences=2)
            if wiki_result and "Multiple results found" not in wiki_result:
                results.append(f"📖 **Wikipedia:**\n{wiki_result}\n")
            
            # Enhance with AI knowledge
            ai_prompt = f"""Provide additional context and insights about "{query}" that complements encyclopedia information. Focus on:
- Practical applications
- Recent developments
- Why it matters
- Common misconceptions

Keep it concise (2-3 paragraphs)."""
            
            ai_result = ollama_manager.chat_with_memory(ai_prompt)
            results.append(f"🤖 **AI Insights:**\n{ai_result}")
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Smart search error: {e}")
            return f"❌ Search failed: {e}"
    
    # ============ YouTube Search ============
    
    def open_youtube(self, query: str) -> str:
        """
        Open YouTube search in browser
        
        Args:
            query: Search query
            
        Returns:
            Confirmation message
        """
        try:
            url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            webbrowser.open(url)
            logger.info(f"Opened YouTube search for: {query}")
            return f"▶️ Opening YouTube search for '{query}'..."
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
            return f"❌ Failed to open YouTube search: {e}"
    
    # ============ Weather Service (AI-Powered) ============
    
    def get_weather(self, city: str) -> Optional[str]:
        """
        Get weather information using AI-powered web search
        
        Args:
            city: City name
            
        Returns:
            Formatted weather information or None
        """
        try:
            logger.info(f"Getting weather for: {city}")
            
            # Use AI web search to get current weather
            query = f"current weather in {city} today temperature humidity wind conditions"
            weather_info = self.ai_web_search(query)
            
            if weather_info:
                # Format the response nicely
                formatted = f"🌤️ **Weather in {city}:**\n\n{weather_info}"
                logger.info(f"Weather information retrieved for: {city}")
                return formatted
            else:
                logger.warning(f"Could not retrieve weather for: {city}")
                return f"❌ Could not retrieve weather information for {city}. Please try again."
                
        except Exception as e:
            logger.error(f"Weather retrieval error: {e}")
            return f"❌ Error getting weather information: {str(e)}"
    
    # ============ News Service (AI-Powered) ============
    
    def get_news(self, country: str = "us", category: str = None) -> Optional[str]:
        """
        Get latest news headlines using AI-powered web search
        
        Args:
            country: Country code (us, in, gb, etc.)
            category: News category (optional)
            
        Returns:
            Formatted news headlines or None
        """
        try:
            logger.info(f"Getting news for country: {country}, category: {category}")
            
            # Build query based on parameters
            country_names = {
                "us": "United States",
                "in": "India",
                "gb": "United Kingdom",
                "ca": "Canada",
                "au": "Australia"
            }
            country_name = country_names.get(country.lower(), country)
            
            query = f"latest news headlines today in {country_name}"
            if category:
                query = f"latest {category} news today in {country_name}"
            
            # Use AI web search to get news
            news_info = self.ai_web_search(query)
            
            if news_info:
                formatted = f"📰 **Latest News ({country_name}):**\n\n{news_info}"
                logger.info(f"News headlines retrieved for: {country_name}")
                return formatted
            else:
                logger.warning(f"Could not retrieve news for: {country_name}")
                return f"❌ Could not retrieve news for {country_name}. Please try again."
                
        except Exception as e:
            logger.error(f"News retrieval error: {e}")
            return f"❌ Error getting news: {str(e)}"
    
    def get_news_by_topic(self, topic: str, count: int = 5) -> Optional[str]:
        """
        Search news by specific topic using AI-powered web search
        
        Args:
            topic: News topic to search for
            count: Number of articles to retrieve (informational, AI decides)
            
        Returns:
            Formatted news headlines or None
        """
        try:
            logger.info(f"Searching news for topic: {topic}")
            
            # Use AI web search for topic-specific news
            query = f"latest news about {topic} today current events"
            news_info = self.ai_web_search(query)
            
            if news_info:
                formatted = f"📰 **News about {topic}:**\n\n{news_info}"
                logger.info(f"Topic news retrieved for: {topic}")
                return formatted
            else:
                logger.warning(f"Could not retrieve news for topic: {topic}")
                return f"❌ Could not find news about {topic}. Please try another topic."
                
        except Exception as e:
            logger.error(f"Topic news retrieval error: {e}")
            return f"❌ Error getting news about {topic}: {str(e)}"
    
    # ============ Advanced Search Features ============
    
    def web_search_with_summary(self, query: str, model: str = None) -> str:
        """
        Perform web search and provide AI-generated summary
        
        Args:
            query: Search query
            model: Ollama model to use
            
        Returns:
            Search results with summary
        """
        try:
            # Try multiple sources
            results = []
            
            # Wikipedia
            wiki_result = self.search_wikipedia(query, sentences=3)
            if wiki_result and "Multiple results found" not in wiki_result:
                results.append(f"📖 **From Wikipedia:**\n{wiki_result}")
            
            # AI-powered search
            if ollama_manager:
                search_prompt = f"""Search the web for: "{query}"

Provide:
1. Key information and facts
2. Most relevant details
3. Recent updates or news (if applicable)
4. Practical insights

Be concise but comprehensive."""
                
                ai_response = ollama_manager.chat_with_memory(search_prompt, model_name=model)
                results.append(f"\n🤖 **AI Search Analysis:**\n{ai_response}")
            
            # News (if topic-related)
            news_results = self.get_news_by_topic(query, count=3)
            if news_results:
                results.append(f"\n📰 **Related News:**\n" + "\n".join(news_results))
            
            return "\n".join(results) if results else "No results found."
            
        except Exception as e:
            logger.error(f"Web search with summary error: {e}")
            return f"❌ Search failed: {e}"
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """
        Get AI-generated search suggestions
        
        Args:
            query: Original search query
            
        Returns:
            List of suggested search terms
        """
        try:
            prompt = f"""Given the search query: "{query}"

Suggest 5 related search terms that would help the user find more specific or relevant information.

Respond with only the search terms, one per line, without numbering or explanations."""
            
            response = ollama_manager.chat_with_memory(prompt)
            suggestions = [s.strip() for s in response.split('\n') if s.strip()]
            
            return suggestions[:5]  # Limit to 5
            
        except Exception as e:
            logger.error(f"Search suggestions error: {e}")
            return []


# Create global instance
web_services = WebServices()


# Convenience functions for backward compatibility
def search_wikipedia(topic: str) -> Optional[str]:
    """Search Wikipedia - convenience function"""
    return web_services.search_wikipedia(topic)


def open_google(query: str) -> str:
    """Open Google search - convenience function"""
    return web_services.open_google(query)


def open_youtube(query: str) -> str:
    """Open YouTube search - convenience function"""
    return web_services.open_youtube(query)


def get_weather(city: str) -> Optional[Dict[str, Any]]:
    """Get weather - convenience function"""
    return web_services.get_weather(city)


def get_news(country: str = "us") -> Optional[List[str]]:
    """Get news - convenience function"""
    return web_services.get_news(country)


def format_weather_response(weather_data: Dict[str, Any]) -> str:
    """Format weather - convenience function"""
    return web_services.format_weather_response(weather_data)
