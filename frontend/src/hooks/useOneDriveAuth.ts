import { useEffect, useRef } from "react";

const ONE_DRIVE_POPUP_URL = "/api/auth/start";

type LogType = "info" | "success" | "error" | "warning";

export function useOneDriveAuth(
  setIsConnected: (v: boolean) => void,
  addLog: (message: string, type?: LogType) => void,
) {
  const restoredRef = useRef(false);

  const checkAuthStatus = () => {
    return fetch("/api/auth/me", { credentials: "same-origin" })
      .then((r) => r.json())
      .then((data) => {
        setIsConnected(!!data.authenticated);
        return data.authenticated as boolean;
      })
      .catch(() => false);
  };

  // On mount: check if MSAL cache has a valid account
  useEffect(() => {
    if (!restoredRef.current) {
      restoredRef.current = true;
      checkAuthStatus().then((authenticated) => {
        if (authenticated) addLog("Restored OneDrive session", "success");
      });
    }
  }, []);

  const connect = () => {
    addLog("Opening OneDrive auth…");
    const popup = window.open(ONE_DRIVE_POPUP_URL, "authPopup", "width=600,height=700");
    if (!popup) {
      addLog("Popup was blocked — please allow popups for this site.", "error");
      return;
    }
    // Poll for popup close, then verify with backend
    const poll = setInterval(() => {
      if (popup.closed) {
        clearInterval(poll);
        checkAuthStatus().then((authenticated) => {
          if (authenticated) addLog("OneDrive connected", "success");
          else addLog("Authentication failed or was cancelled", "warning");
        });
      }
    }, 500);
  };

  const logout = () => {
    fetch("/api/auth/logout", { method: "POST", credentials: "same-origin" })
      .catch(() => {});
    localStorage.removeItem("access_token");
    setIsConnected(false);
    addLog("Logged out of OneDrive", "info");
  };

  return { connect, logout };
}
