import { createContext, useContext } from 'react';
import type { SessionUser } from './api';

export const Account = createContext<SessionUser | null>(null);
export const useAccount = () => useContext(Account);
