import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Play, Pause, RotateCcw } from 'lucide-react';

interface BreathingGuardianProps {
  onComplete?: (duration: number) => void;
}

type BreathPhase = 'inhale' | 'hold' | 'exhale' | 'pause';

const BreathingGuardian: React.FC<BreathingGuardianProps> = ({ onComplete }) => {
  const [isActive, setIsActive] = useState(false);
  const [phase, setPhase] = useState<BreathPhase>('inhale');
  const [duration, setDuration] = useState(2); // minutes
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [cycleProgress, setCycleProgress] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout>();

  const durations = [2, 5, 10];
  const breathCycle = {
    inhale: 4000,  // 4 seconds
    hold: 2000,    // 2 seconds  
    exhale: 6000,  // 6 seconds
    pause: 1000    // 1 second
  };

  useEffect(() => {
    if (isActive) {
      setTimeRemaining(duration * 60);
      startBreathingCycle();
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isActive, duration]);

  const startBreathingCycle = () => {
    const phases: BreathPhase[] = ['inhale', 'hold', 'exhale', 'pause'];
    let currentPhaseIndex = 0;
    let phaseStartTime = Date.now();
    let totalStartTime = Date.now();

    const updateCycle = () => {
      const now = Date.now();
      const currentPhaseDuration = breathCycle[phases[currentPhaseIndex]];
      const phaseElapsed = now - phaseStartTime;
      const totalElapsed = now - totalStartTime;
      
      // Update time remaining
      const remaining = Math.max(0, (duration * 60 * 1000) - totalElapsed);
      setTimeRemaining(Math.ceil(remaining / 1000));
      
      // Update cycle progress
      setCycleProgress((phaseElapsed / currentPhaseDuration) * 100);
      
      // Check if phase complete
      if (phaseElapsed >= currentPhaseDuration) {
        currentPhaseIndex = (currentPhaseIndex + 1) % phases.length;
        setPhase(phases[currentPhaseIndex]);
        phaseStartTime = now;
        setCycleProgress(0);
      }
      
      // Check if session complete
      if (remaining <= 0) {
        setIsActive(false);
        setPhase('inhale');
        setCycleProgress(0);
        onComplete?.(duration);
        return;
      }
    };

    intervalRef.current = setInterval(updateCycle, 50);
    updateCycle();
  };

  const resetSession = () => {
    setIsActive(false);
    setPhase('inhale');
    setCycleProgress(0);
    setTimeRemaining(0);
  };

  const getPhaseInstruction = () => {
    switch (phase) {
      case 'inhale': return 'Breathe In';
      case 'hold': return 'Hold';
      case 'exhale': return 'Breathe Out';
      case 'pause': return 'Pause';
    }
  };

  const getAnimationClasses = () => {
    if (!isActive) return '';
    
    switch (phase) {
      case 'inhale': return 'animate-breathe-in';
      case 'exhale': return 'animate-breathe-out';
      default: return 'animate-pulse-gentle';
    }
  };

  return (
    <Card className="p-8 bg-gradient-breath shadow-embrace">
      <div className="text-center space-y-8">
        {/* Header */}
        <div>
          <h2 className="text-2xl font-semibold text-foreground mb-2">
            Breathing Guardian
          </h2>
          <p className="text-muted-foreground">
            {isActive ? 'Follow the rhythm and breathe with intention' : 'Find your calm through conscious breathing'}
          </p>
        </div>

        {/* Duration Selection */}
        {!isActive && (
          <div className="flex justify-center gap-3">
            {durations.map((d) => (
              <Button
                key={d}
                variant={duration === d ? "default" : "outline"}
                onClick={() => setDuration(d)}
                className="min-w-16"
              >
                {d}m
              </Button>
            ))}
          </div>
        )}

        {/* Breathing Visualization */}
        <div className="flex flex-col items-center space-y-6">
          <div 
            className={`w-64 h-64 rounded-full bg-gradient-breath border-4 border-primary/20 flex items-center justify-center transition-all duration-300 ${getAnimationClasses()}`}
            style={{
              transform: isActive ? `scale(${1 + (cycleProgress / 300)})` : 'scale(1)',
              opacity: isActive ? 0.6 + (cycleProgress / 250) : 0.8
            }}
          >
            <div className="text-center">
              <div className="text-xl font-semibold text-primary-foreground mb-2">
                {getPhaseInstruction()}
              </div>
              {isActive && (
                <div className="text-sm text-primary-foreground/80">
                  {Math.floor(timeRemaining / 60)}:{(timeRemaining % 60).toString().padStart(2, '0')}
                </div>
              )}
            </div>
          </div>

          {/* Phase Progress */}
          {isActive && (
            <div className="w-64 h-2 bg-primary/20 rounded-full overflow-hidden">
              <div 
                className="h-full bg-primary transition-all duration-75 ease-out rounded-full"
                style={{ width: `${cycleProgress}%` }}
              />
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex justify-center gap-4">
          <Button
            onClick={() => setIsActive(!isActive)}
            size="lg"
            className="px-8"
          >
            {isActive ? <Pause className="w-5 h-5 mr-2" /> : <Play className="w-5 h-5 mr-2" />}
            {isActive ? 'Pause' : 'Start'}
          </Button>
          
          {(isActive || timeRemaining > 0) && (
            <Button
              onClick={resetSession}
              size="lg"
              variant="outline"
            >
              <RotateCcw className="w-5 h-5 mr-2" />
              Reset
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
};

export default BreathingGuardian;