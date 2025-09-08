import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useNavigate } from "react-router-dom";
import { CheckCircle2, Copy, ArrowLeft } from "lucide-react";

const TOKEN_KEY = "access_token";

const AuthSuccess = () => {
  const navigate = useNavigate();
  const [copied, setCopied] = useState(false);

  // 1. Parse the token from the URL hash (e.g. "#token=XYZ")
  const hash = window.location.hash;
  const params = new URLSearchParams(hash.slice(1));
  const accessToken = params.get("token") || "";

  // 2. On mount, store token in the opener, dispatch storage event, notify via postMessage, then close popup
  useEffect(() => {
    if (accessToken) {
      // Write into parent window’s localStorage
      window.opener?.localStorage.setItem(TOKEN_KEY, accessToken);
      // Dispatch a storage event on the opener to trigger any listeners
      window.opener?.dispatchEvent(
        new StorageEvent('storage', { key: TOKEN_KEY, newValue: accessToken })
      );
      // Fallback notification via postMessage
      window.opener?.postMessage({ type: "onedrive_connected" }, "*");
      // Close the popup after a brief delay so the UI renders once
      setTimeout(() => window.close(), 500);
    } else {
      // No token found: send the user back home
      navigate("/", { replace: true });
    }
  }, [accessToken, navigate]);


  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 to-slate-200">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/")}
              className="gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Home
            </Button>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-slate-800">
                LeibnizDream
              </h1>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-16 max-w-2xl">
        <div className="text-center space-y-8">
          {/* Success Icon */}
          <div className="mx-auto p-8 bg-green-100 rounded-full w-fit">
            <CheckCircle2 className="w-16 h-16 text-green-600" />
          </div>

          {/* Success Message */}
          <div className="space-y-4">
            <h2 className="text-3xl font-bold text-slate-800">
              ✅ Signed in successfully
            </h2>
            <p className="text-lg text-slate-600">
              Your access token is ready.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-16 py-8 border-t border-slate-200 bg-white/50">
        <div className="container mx-auto px-4 text-center">
          <div className="flex items-center justify-center gap-2 text-slate-600">
            <p>LeibnizDream</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default AuthSuccess;
