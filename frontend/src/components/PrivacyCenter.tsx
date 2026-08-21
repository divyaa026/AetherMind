import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import MockDataBanner from '@/components/MockDataBanner';
import {
  Shield,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  Database,
  Download,
  Trash2,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Server,
  Smartphone,
  Cloud,
  CloudOff,
  HardDrive,
  Key,
  FileText,
  Settings,
  ChevronRight,
  Info
} from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import AetherMindAPIService, { PrivacySettings, DataCollection, FederatedStatus } from '@/services/AetherMindAPI';

interface PrivacyCenterProps {
  onNavigate?: (tab: string) => void;
}

const PrivacyCenter: React.FC<PrivacyCenterProps> = ({ onNavigate }) => {
  const [settings, setSettings] = useState<PrivacySettings | null>(null);
  const [dataCollection, setDataCollection] = useState<DataCollection[]>([]);
  const [federatedStatus, setFederatedStatus] = useState<FederatedStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [deletingData, setDeletingData] = useState(false);

  useEffect(() => {
    loadPrivacyData();
  }, []);

  const loadPrivacyData = async () => {
    setLoading(true);
    try {
      const [settingsData, collectionData, fedStatus] = await Promise.all([
        AetherMindAPIService.getPrivacySettings(),
        AetherMindAPIService.getDataCollection(),
        AetherMindAPIService.getFederatedLearningStatus()
      ]);
      setSettings(settingsData);
      setDataCollection(collectionData);
      setFederatedStatus(fedStatus);
    } catch (error) {
      console.error('Failed to load privacy data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTogglePermission = async (permissionId: string, enabled: boolean) => {
    try {
      await AetherMindAPIService.updatePrivacyPermission(permissionId, enabled);
      setDataCollection(dataCollection.map(d =>
        d.id === permissionId ? { ...d, enabled } : d
      ));
      toast({
        title: enabled ? "Permission enabled" : "Permission disabled",
        description: "Your privacy settings have been updated.",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to update settings",
        variant: "destructive"
      });
    }
  };

  const handleDeleteAllData = async () => {
    if (!confirm('Are you sure you want to delete all your data? This action cannot be undone.')) {
      return;
    }
    
    setDeletingData(true);
    try {
      await AetherMindAPIService.deleteAllData();
      toast({
        title: "Data deleted",
        description: "All your data has been permanently removed.",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete data",
        variant: "destructive"
      });
    } finally {
      setDeletingData(false);
    }
  };

  const handleExportData = async () => {
    try {
      toast({
        title: "Preparing export...",
        description: "Your data package is being generated.",
      });
      await AetherMindAPIService.exportAllData();
      toast({
        title: "Export ready!",
        description: "Your data package is downloading.",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to export data",
        variant: "destructive"
      });
    }
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-muted rounded w-1/3"></div>
          <div className="h-48 bg-muted rounded"></div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <MockDataBanner feature="Privacy Center" storageType="static" />
      
      {/* Header */}
      <Card className="p-6 bg-gradient-to-br from-primary/10 via-secondary/5 to-background">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Shield className="w-6 h-6 text-primary" />
            <div>
              <h2 className="text-xl font-semibold">Privacy Control Center</h2>
              <p className="text-sm text-muted-foreground">
                Your data, your control. Complete transparency.
              </p>
            </div>
          </div>
        </div>

        {/* Privacy Score */}
        {settings && (
          <div className="flex items-center gap-4 p-4 bg-success/10 rounded-lg border border-success/20">
            <div className="w-16 h-16 rounded-full bg-success/20 flex items-center justify-center">
              <Lock className="w-8 h-8 text-success" />
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Privacy Score</span>
                <span className="text-2xl font-bold text-success">{settings.privacyScore}/100</span>
              </div>
              <Progress value={settings.privacyScore} className="h-2" />
            </div>
          </div>
        )}
      </Card>

      {/* Data Dashboard */}
      <Card className="p-6">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-primary" />
          What We Collect
        </h3>
        <p className="text-sm text-muted-foreground mb-6">
          Here's exactly what data we store and why. Toggle any permission on or off.
        </p>

        <div className="space-y-4">
          {dataCollection.map((item) => (
            <div 
              key={item.id}
              className={`p-4 rounded-lg border transition-colors ${
                item.enabled ? 'bg-card border-border' : 'bg-muted/50 border-muted'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                    item.enabled ? 'bg-primary/10 text-primary' : 'bg-muted text-muted-foreground'
                  }`}>
                    {item.icon === 'smartphone' && <Smartphone className="w-5 h-5" />}
                    {item.icon === 'file-text' && <FileText className="w-5 h-5" />}
                    {item.icon === 'database' && <Database className="w-5 h-5" />}
                    {item.icon === 'eye' && <Eye className="w-5 h-5" />}
                    {item.icon === 'server' && <Server className="w-5 h-5" />}
                  </div>
                  
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium">{item.name}</span>
                      {item.required && (
                        <Badge variant="outline" className="text-xs">Required</Badge>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      {item.description}
                    </p>
                    <div className="text-xs text-muted-foreground">
                      Purpose: {item.purpose}
                    </div>
                  </div>
                </div>

                <Switch
                  checked={item.enabled}
                  disabled={item.required}
                  onCheckedChange={(checked) => handleTogglePermission(item.id, checked)}
                />
              </div>

              {/* Data Stats */}
              {item.enabled && item.stats && (
                <div className="mt-3 pt-3 border-t flex items-center gap-4 text-xs text-muted-foreground">
                  <span>Records: {item.stats.records}</span>
                  <span>Size: {item.stats.size}</span>
                  <span>Last updated: {item.stats.lastUpdated}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Federated Learning Status */}
      {federatedStatus && (
        <Card className="p-6">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Server className="w-5 h-5 text-secondary" />
            Federated Learning Status
          </h3>
          
          <div className="space-y-4">
            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-3 h-3 rounded-full bg-success animate-pulse" />
                <span className="font-medium">Personal Model</span>
                <Badge variant="secondary">Trained Locally</Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                Your personal AI model is trained entirely on your device. 
                Raw data never leaves your phone.
              </p>
            </div>

            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-3 h-3 rounded-full bg-primary" />
                <span className="font-medium">Aggregate Updates</span>
                <Badge variant="outline">Anonymous</Badge>
              </div>
              <p className="text-sm text-muted-foreground mb-3">
                Only encrypted model improvements are shared, not your data. 
                These updates help improve the system for everyone.
              </p>
              
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Contribute anonymous updates</span>
                <Switch defaultChecked />
              </div>
            </div>

            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-3 mb-3">
                <HardDrive className="w-5 h-5 text-muted-foreground" />
                <span className="font-medium">On-Device Storage</span>
              </div>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="font-semibold">{federatedStatus.modelSize}</div>
                  <div className="text-xs text-muted-foreground">Model size</div>
                </div>
                <div>
                  <div className="font-semibold">{federatedStatus.lastTraining}</div>
                  <div className="text-xs text-muted-foreground">Last training</div>
                </div>
                <div>
                  <div className="font-semibold">{federatedStatus.dataPoints}</div>
                  <div className="text-xs text-muted-foreground">Data points</div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Data Management */}
      <Card className="p-6">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5 text-primary" />
          Data Management
        </h3>

        <div className="space-y-3">
          <Button 
            variant="outline" 
            className="w-full justify-between h-auto py-4"
            onClick={handleExportData}
          >
            <div className="flex items-center gap-3">
              <Download className="w-5 h-5" />
              <div className="text-left">
                <div className="font-medium">Export All Data</div>
                <div className="text-xs text-muted-foreground">
                  Download a complete copy of your data
                </div>
              </div>
            </div>
            <ChevronRight className="w-5 h-5" />
          </Button>

          <Button 
            variant="outline" 
            className="w-full justify-between h-auto py-4"
          >
            <div className="flex items-center gap-3">
              <RefreshCw className="w-5 h-5" />
              <div className="text-left">
                <div className="font-medium">Clear Cache</div>
                <div className="text-xs text-muted-foreground">
                  Remove temporary files (keeps your data safe)
                </div>
              </div>
            </div>
            <ChevronRight className="w-5 h-5" />
          </Button>
        </div>
      </Card>

      {/* Danger Zone */}
      <Card className="p-6 border-destructive/20">
        <h3 className="font-semibold mb-4 flex items-center gap-2 text-destructive">
          <AlertTriangle className="w-5 h-5" />
          Danger Zone
        </h3>

        <Alert variant="destructive" className="mb-4">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            These actions are permanent and cannot be undone.
          </AlertDescription>
        </Alert>

        <Button 
          variant="destructive"
          className="w-full"
          onClick={handleDeleteAllData}
          disabled={deletingData}
        >
          {deletingData ? (
            <>
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              Deleting...
            </>
          ) : (
            <>
              <Trash2 className="w-4 h-4 mr-2" />
              Delete All Data Instantly
            </>
          )}
        </Button>

        <p className="text-xs text-muted-foreground mt-3 text-center">
          This will permanently delete all your check-ins, journal entries, 
          progress data, and personal settings.
        </p>
      </Card>

      {/* Info Card */}
      <Card className="p-4 bg-primary/5 border-primary/20">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-primary mt-0.5" />
          <div className="text-sm">
            <p className="font-medium mb-1">Our Privacy Promise</p>
            <p className="text-muted-foreground">
              AetherMind is built with privacy-first architecture. Your mental health data 
              is encrypted, stored locally when possible, and never sold to third parties.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default PrivacyCenter;
