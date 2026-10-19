import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch, apiPost } from './client';
import { ChangeEvent } from '../types';

export function useReviewQueue() {
  return useQuery<ChangeEvent[]>({
    queryKey: ['reviewQueue'],
    queryFn: () => apiFetch<ChangeEvent[]>('/api/review-queue'),
    staleTime: 1000 * 30,
    refetchInterval: 1000 * 60,
  });
}

export function useSubmitDecision() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      eventId,
      analystLabel,
      analystId,
      note,
    }: {
      eventId: string;
      analystLabel: 'confirm' | 'reject';
      analystId: string;
      note?: string;
    }) =>
      apiPost(`/api/review/${eventId}/decision`, {
        analyst_label: analystLabel,
        analyst_id: analystId,
        note: note ?? '',
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['reviewQueue'] });
    },
  });
}

export function useRunChangeAnalysis() {
  return useMutation({
    mutationFn: ({
      startDate,
      endDate,
      bbox,
    }: {
      startDate: string;
      endDate: string;
      bbox: number[];
    }) =>
      apiPost('/api/change/analyze', {
        start_date: startDate,
        end_date: endDate,
        bbox,
      }),
  });
}

export function useCluster(tileId: string | null) {
  return useQuery({
    queryKey: ['cluster', tileId],
    queryFn: () => apiFetch(`/api/cluster?tile_id=${tileId}`),
    enabled: !!tileId,
    staleTime: 1000 * 60 * 5,
  });
}

export function useAuditTrail() {
  return useQuery({
    queryKey: ['audit'],
    queryFn: () => apiFetch('/api/audit'),
    staleTime: 1000 * 60,
  });
}
