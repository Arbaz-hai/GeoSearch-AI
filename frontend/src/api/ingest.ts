import { useMutation } from '@tanstack/react-query';
import { apiPost } from './client';

export function useIngestScene() {
  return useMutation({
    mutationFn: (formData: FormData) =>
      fetch('/api/ingest/scene', {
        method: 'POST',
        body: formData,
      }).then((r) => {
        if (!r.ok) throw new Error(`Ingest error ${r.status}`);
        return r.json();
      }),
  });
}
