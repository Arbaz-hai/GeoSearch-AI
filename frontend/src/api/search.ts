import { useQuery } from '@tanstack/react-query';
import { apiFetch, apiPost } from './client';
import { SearchResult } from '../types';

interface SearchFilters {
  sensor?: string;
  date_start?: string;
  date_end?: string;
  bbox?: number[];
  limit?: number;
}

export function useTextSearch(query: string, filters: SearchFilters, enabled: boolean) {
  return useQuery<SearchResult[]>({
    queryKey: ['textSearch', query, filters],
    queryFn: () =>
      apiPost<SearchResult[]>('/api/search/text', {
        query,
        ...filters,
        limit: filters.limit ?? 20,
      }),
    enabled: enabled && !!query,
    staleTime: 1000 * 60,
  });
}

export function useLibrary() {
  return useQuery<{ scenes: unknown[] }>({
    queryKey: ['library'],
    queryFn: () => apiFetch('/api/library'),
    staleTime: 1000 * 60 * 5,
  });
}
