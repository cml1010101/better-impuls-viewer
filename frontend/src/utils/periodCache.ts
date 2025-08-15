// Client-side cache for storing periods per star
interface CachedPeriod {
  period: number | null; // null represents "no data" case
  timestamp: number;
  campaign: {
    starNumber: number;
    survey: string;
    campaignId: number;
  };
  isPrimary?: boolean;
  category?: string; // Light curve category classification
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
    const cached = cache[key];
    return cached?.period ?? null;
  }

  static setCachedPeriod(starNumber: number, survey: string, campaignId: number, period: number | null, isPrimary?: boolean, category?: string): void {
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
        },
        isPrimary: isPrimary || false,
        category
      };
      
      localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
    } catch (error) {
      console.error('Error saving period to cache:', error);
    }
  }

  static getAllCachedPeriodsForStar(starNumber: number): Array<{
    survey: string;
    campaignId: number;
    period: number | null;
    timestamp: number;
    isPrimary: boolean;
    category?: string;
  }> {
    const cache = this.getCachedPeriods();
    return Object.values(cache)
      .filter(cached => cached.campaign.starNumber === starNumber)
      .map(cached => ({
        survey: cached.campaign.survey,
        campaignId: cached.campaign.campaignId,
        period: cached.period,
        timestamp: cached.timestamp,
        isPrimary: cached.isPrimary || false,
        category: cached.category
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

  static setPrimaryPeriod(starNumber: number, survey: string, campaignId: number): void {
    try {
      const cache = this.getCachedPeriods();
      
      // Clear existing primary for this star
      for (const [, value] of Object.entries(cache)) {
        if (value.campaign.starNumber === starNumber) {
          value.isPrimary = false;
        }
      }
      
      // Set new primary
      const key = this.getCacheKey(starNumber, survey, campaignId);
      if (cache[key]) {
        cache[key].isPrimary = true;
        localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
      }
    } catch (error) {
      console.error('Error setting primary period:', error);
    }
  }

  static getPrimaryPeriodForStar(starNumber: number): {
    survey: string;
    campaignId: number;
    period: number | null;
    timestamp: number;
  } | null {
    const cache = this.getCachedPeriods();
    const primaryEntry = Object.values(cache).find(
      cached => cached.campaign.starNumber === starNumber && cached.isPrimary
    );
    
    if (primaryEntry) {
      return {
        survey: primaryEntry.campaign.survey,
        campaignId: primaryEntry.campaign.campaignId,
        period: primaryEntry.period,
        timestamp: primaryEntry.timestamp
      };
    }
    
    return null;
  }

  static getAllPrimaryPeriods(): Array<{
    starNumber: number;
    survey: string;
    campaignId: number;
    period: number | null;
    timestamp: number;
  }> {
    const cache = this.getCachedPeriods();
    return Object.values(cache)
      .filter(cached => cached.isPrimary)
      .map(cached => ({
        starNumber: cached.campaign.starNumber,
        survey: cached.campaign.survey,
        campaignId: cached.campaign.campaignId,
        period: cached.period,
        timestamp: cached.timestamp
      }));
  }

  static setCachedPeriodCategory(starNumber: number, survey: string, campaignId: number, category: string): void {
    try {
      const cache = this.getCachedPeriods();
      const key = this.getCacheKey(starNumber, survey, campaignId);
      
      if (cache[key]) {
        cache[key].category = category;
        localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
      }
    } catch (error) {
      console.error('Error setting period category:', error);
    }
  }

  static removeCachedPeriod(starNumber: number, survey: string, campaignId: number): void {
    try {
      const cache = this.getCachedPeriods();
      const key = this.getCacheKey(starNumber, survey, campaignId);
      
      delete cache[key];
      localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
    } catch (error) {
      console.error('Error removing cached period:', error);
    }
  }

  static exportToCSV(): string {
    const cache = this.getCachedPeriods();
    const entries = Object.values(cache);
    
    if (entries.length === 0) {
      return 'star_number,survey,campaign_id,period,category,is_primary,timestamp\n';
    }

    const header = 'star_number,survey,campaign_id,period,category,is_primary,timestamp\n';
    const rows = entries.map(entry => {
      const period = entry.period === null || entry.period === undefined ? -9 : entry.period;
      const category = entry.category || 'unknown';
      const isPrimary = entry.isPrimary ? 'true' : 'false';
      
      return `${entry.campaign.starNumber},${entry.campaign.survey},${entry.campaign.campaignId},${period},${category},${isPrimary},${entry.timestamp}`;
    }).join('\n');

    return header + rows;
  }

  static importFromCSV(csvContent: string, overwrite: boolean = false): { success: boolean; imported: number; errors: string[] } {
    const lines = csvContent.trim().split('\n');
    const errors: string[] = [];
    let imported = 0;

    if (lines.length < 2) {
      return { success: false, imported: 0, errors: ['CSV file is empty or missing header'] };
    }

    const header = lines[0].toLowerCase();
    const expectedHeaders = ['star_number', 'survey', 'campaign_id', 'period', 'category', 'is_primary', 'timestamp'];
    
    for (const expectedHeader of expectedHeaders) {
      if (!header.includes(expectedHeader)) {
        errors.push(`Missing required column: ${expectedHeader}`);
      }
    }

    if (errors.length > 0) {
      return { success: false, imported: 0, errors };
    }

    const cache = overwrite ? {} : this.getCachedPeriods();

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      try {
        const columns = line.split(',');
        if (columns.length < 7) {
          errors.push(`Line ${i + 1}: Insufficient columns`);
          continue;
        }

        const starNumber = parseInt(columns[0]);
        const survey = columns[1];
        const campaignId = parseInt(columns[2]);
        const period = parseFloat(columns[3]);
        const category = columns[4];
        const isPrimary = columns[5].toLowerCase() === 'true';
        const timestamp = parseInt(columns[6]) || Date.now();

        if (isNaN(starNumber) || isNaN(campaignId)) {
          errors.push(`Line ${i + 1}: Invalid star number or campaign ID`);
          continue;
        }

        // Handle "no data" case (period = -9)
        const actualPeriod = period === -9 ? null : period;
        if (actualPeriod !== null && (isNaN(actualPeriod) || actualPeriod < 0.1 || actualPeriod > 100)) {
          errors.push(`Line ${i + 1}: Invalid period value: ${period}`);
          continue;
        }

        const key = this.getCacheKey(starNumber, survey, campaignId);
        cache[key] = {
          period: actualPeriod,
          timestamp,
          campaign: {
            starNumber,
            survey,
            campaignId
          },
          isPrimary,
          category: category !== 'unknown' ? category : undefined
        };

        imported++;
      } catch (error) {
        errors.push(`Line ${i + 1}: Parse error - ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }

    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
      return { success: true, imported, errors };
    } catch {
      return { success: false, imported: 0, errors: ['Failed to save imported data to cache'] };
    }
  }

  static downloadCSV(filename: string = 'period_cache.csv'): void {
    const csvContent = this.exportToCSV();
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', filename);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }
}

export default PeriodCache;