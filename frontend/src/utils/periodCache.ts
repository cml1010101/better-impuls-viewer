// Client-side cache for storing periods per star
interface CachedPeriod {
  period: number;
  timestamp: number;
  campaign: {
    starNumber: number;
    survey: string;
    campaignId: number;
  };
}

const CACHE_KEY = 'star_periods_cache';
const CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24 hours in milliseconds

export class PeriodCache {
  private static getCacheKey(starNumber: number, survey: string, campaignId: number): string {
    return `${starNumber}_${survey}_${campaignId}`;
  }

  static getCachedPeriods(): Record<string, CachedPeriod> {
    try {
      const cached = localStorage.getItem(CACHE_KEY);
      if (!cached) return {};
      
      const parsed = JSON.parse(cached);
      const now = Date.now();
      
      // Filter out expired entries
      const validEntries: Record<string, CachedPeriod> = {};
      for (const [key, value] of Object.entries(parsed)) {
        const cachedPeriod = value as CachedPeriod;
        if (now - cachedPeriod.timestamp < CACHE_EXPIRY) {
          validEntries[key] = cachedPeriod;
        }
      }
      
      // Save filtered cache back to localStorage
      localStorage.setItem(CACHE_KEY, JSON.stringify(validEntries));
      return validEntries;
    } catch (error) {
      console.error('Error reading period cache:', error);
      return {};
    }
  }

  static getCachedPeriod(starNumber: number, survey: string, campaignId: number): number | null {
    const cache = this.getCachedPeriods();
    const key = this.getCacheKey(starNumber, survey, campaignId);
    return cache[key]?.period || null;
  }

  static setCachedPeriod(starNumber: number, survey: string, campaignId: number, period: number): void {
    try {
      const cache = this.getCachedPeriods();
      const key = this.getCacheKey(starNumber, survey, campaignId);
      
      cache[key] = {
        period,
        timestamp: Date.now(),
        campaign: {
          starNumber,
          survey,
          campaignId
        }
      };
      
      localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
    } catch (error) {
      console.error('Error saving period to cache:', error);
    }
  }

  static getAllCachedPeriodsForStar(starNumber: number): Array<{
    survey: string;
    campaignId: number;
    period: number;
    timestamp: number;
  }> {
    const cache = this.getCachedPeriods();
    return Object.values(cache)
      .filter(cached => cached.campaign.starNumber === starNumber)
      .map(cached => ({
        survey: cached.campaign.survey,
        campaignId: cached.campaign.campaignId,
        period: cached.period,
        timestamp: cached.timestamp
      }));
  }

  static clearCache(): void {
    try {
      localStorage.removeItem(CACHE_KEY);
    } catch (error) {
      console.error('Error clearing period cache:', error);
    }
  }

  static clearCacheForStar(starNumber: number): void {
    try {
      const cache = this.getCachedPeriods();
      const filteredCache: Record<string, CachedPeriod> = {};
      
      for (const [key, value] of Object.entries(cache)) {
        if (value.campaign.starNumber !== starNumber) {
          filteredCache[key] = value;
        }
      }
      
      localStorage.setItem(CACHE_KEY, JSON.stringify(filteredCache));
    } catch (error) {
      console.error('Error clearing cache for star:', error);
    }
  }
}

export default PeriodCache;