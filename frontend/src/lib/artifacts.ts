// Client for the canvas API. Steps are append-only server-side, so an edit is a
// new version and a restore is a new version carrying old text — nothing here
// can destroy history.
import { json, postJson } from './api';
import type { Artifact, ArtifactKind, ArtifactSummary } from './types';

const artifactPath = (id: string) => '/api/artifacts/' + encodeURIComponent(id);

export const listArtifacts = async (threadId: string): Promise<ArtifactSummary[]> =>
  (
    await json<{ artifacts: ArtifactSummary[] }>(
      '/api/threads/' + encodeURIComponent(threadId) + '/artifacts',
    )
  ).artifacts;

export const createArtifact = async (
  threadId: string,
  title: string,
  kind: ArtifactKind,
  content: string,
): Promise<Artifact> =>
  (
    await postJson<{ artifact: Artifact }>(
      '/api/threads/' + encodeURIComponent(threadId) + '/artifacts',
      { title, kind, content },
    )
  ).artifact;

/** Without `version` this is the current version; with it, that one's text. */
export const loadArtifact = async (id: string, version?: number): Promise<Artifact> =>
  (
    await json<{ artifact: Artifact }>(
      artifactPath(id) + (version === undefined ? '' : '?version=' + version),
    )
  ).artifact;

export const saveVersion = async (
  id: string,
  kind: ArtifactKind,
  content: string,
): Promise<Artifact> =>
  (
    await postJson<{ artifact: Artifact }>(artifactPath(id) + '/versions', { kind, content })
  ).artifact;

export const restoreVersion = async (id: string, version: number): Promise<Artifact> =>
  (await postJson<{ artifact: Artifact }>(artifactPath(id) + '/restore', { version }))
    .artifact;

export const deleteArtifact = async (id: string): Promise<void> => {
  await json<{ ok: boolean }>(artifactPath(id), { method: 'DELETE' });
};
