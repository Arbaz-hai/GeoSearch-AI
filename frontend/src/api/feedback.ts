import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiPost } from './client';

export function useSubmitFeedback() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      eventId,
      label,
      analystId,
      note,
    }: {
      eventId: string;
      label: 'confirm' | 'reject' | 'relabel';
      analystId: string;
      note?: string;
    }) =>
      apiPost(`/api/review/${eventId}/decision`, {
        analyst_label: label,
        analyst_id: analystId,
        note: note ?? '',
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['reviewQueue'] });
      queryClient.invalidateQueries({ queryKey: ['audit'] });
    },
  });
}
