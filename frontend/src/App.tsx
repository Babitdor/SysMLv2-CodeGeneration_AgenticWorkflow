import { useEffect, useState } from 'react';
import { Attachments } from './components/Attachments';
import { AuthGate } from './components/AuthGate';
import { Canvas } from './components/Canvas';
import { Composer } from './components/Composer';
import { ModelPanel } from './components/ModelPanel';
import { OutputPane } from './components/OutputPane';
import { SettingsDialog } from './components/SettingsDialog';
import { Sidebar, TopStrip } from './components/Sidebar';
import { StatusBar } from './components/StatusBar';
import { ThreadView } from './components/ThreadView';
import { Visualizer } from './components/Visualizer';
import { useMediaQuery } from './lib/useMediaQuery';
import { WorkspaceProvider, useWorkspace } from './state/WorkspaceContext';

const STORE_KEY = 'sysml.sidebar';
const NARROW = '(max-width: 900px)';

function readRail(): boolean {
  try {
    return localStorage.getItem(STORE_KEY) === 'rail';
  } catch {
    return false;
  }
}

function writeRail(rail: boolean): void {
  try {
    localStorage.setItem(STORE_KEY, rail ? 'rail' : 'open');
  } catch {
    /* private mode: the layout just will not survive a reload */
  }
}

function Shell() {
  const { state } = useWorkspace();
  const narrow = useMediaQuery(NARROW);
  const [rail, setRail] = useState(readRail);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showDocs, setShowDocs] = useState(false);

  useEffect(() => {
    writeRail(rail);
  }, [rail]);

  return (
    <div className="relative flex h-dvh flex-col overflow-hidden">
      <div
        aria-hidden
        className="blueprint pointer-events-none fixed inset-0 z-0 opacity-90"
      />
      <div className="relative z-10 flex min-h-0 flex-1 flex-col">
        <StatusBar onOpenSettings={() => setShowSettings(true)} />
        <div className="flex min-h-0 flex-1">
          <Sidebar
            narrow={narrow}
            rail={rail}
            drawerOpen={drawerOpen}
            onToggleRail={() => setRail((value) => !value)}
            onCloseDrawer={() => setDrawerOpen(false)}
            onOpenSettings={() => setShowSettings(true)}
            onOpenAttachments={() => setShowDocs(true)}
          />
          <main className={'flex min-h-0 min-w-0 flex-1 ' + ((!state.visualizer && !state.model && (state.canvas || state.liveDocument)) ? 'flex-col lg:flex-row' : '')}>
            <div className="flex min-w-0 flex-1 flex-col">
              <TopStrip
                narrow={narrow}
                drawerOpen={drawerOpen}
                onOpenDrawer={() => setDrawerOpen(true)}
              />
              <ThreadView />
              <Composer onOpenAttachments={() => setShowDocs(true)} />
            </div>
            {/* Keyed by turn: opening another card starts a fresh panel. The
                output pane is the resting state — the canvas, the visualizer and
                the model index take the same slot, one at a time. */}
            {state.visualizer ? (
              <Visualizer key={state.visualizer.code} />
            ) : state.model ? (
              <ModelPanel key={state.model.code} />
            ) : state.canvas || state.liveDocument ? (
              <Canvas />
            ) : (
              <OutputPane />
            )}
          </main>
        </div>
      </div>

      {showSettings && <SettingsDialog onClose={() => setShowSettings(false)} />}
      {showDocs && <Attachments onClose={() => setShowDocs(false)} />}
    </div>
  );
}

export default function App() {
  // The gate sits outside the workspace provider: nothing inside it can load
  // before there is a session, and a later 401 sends the user back to the form.
  return (
    <AuthGate>
      <WorkspaceProvider>
        <Shell />
      </WorkspaceProvider>
    </AuthGate>
  );
}
