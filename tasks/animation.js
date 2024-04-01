
//
// Animation and Event objects
//

EVENT_TYPE_INSTANTANEOUS = "instantaneous"
EVENT_TYPE_EXTENDED = "extended"

class Animation {
  constructor(duration, eventSequences) {
    this.duration = duration;
    this.eventSequences = eventSequences;
  }
}

class Event {
  constructor(startTime, type) {
    this.startTime = startTime;
    this.type = type;
  }
}

class InstantaneousEvent extends Event {
  constructor(startTime, occur) {
    super(startTime, EVENT_TYPE_INSTANTANEOUS)
    this.occur = occur;
  }
}

class ExtendedEvent extends Event {
  constructor(startTime, duration, start, update, stop) {
    super(startTime, EVENT_TYPE_EXTENDED)
    this.duration = duration;
    this.start = start;
    this.update = update;
    this.stop = stop;
  }
}

function concatenateAnimations(animations, delay = 0) {
  /* Returns a new animation by concatenating the given animations with the
  given delay in between. */
  let newEventSequences = [];
  let offset = 0;
  for (let animation of animations) {
    for (let eventSequence of animation.eventSequences) {
      let newEventSequence = [];
      for (let event of eventSequence) {
        let newEvent = shallowCopy(event);
        newEvent.startTime += offset;
        newEventSequence.push(newEvent);
      }
      newEventSequences.push(newEventSequence);
    }
    offset += animation.duration + delay;
  }

  const newDuration = (animations.length > 0 ?
    (offset - delay) : 0);
  const newAnimation = new Animation(newDuration, newEventSequences);
  return newAnimation;
}

//
// Playback
//

const PLAYBACK_STATE_READY = 1;
const PLAYBACK_STATE_STARTING = 2;
const PLAYBACK_STATE_PLAYING = 3;
// const PLAYBACK_STATE_PAUSED = 4;
const PLAYBACK_STATE_STOPPING = 5;
const PLAYBACK_STATE_STOPPED_READY_REPLAY = 6;

class AnimationPlayback {
  constructor(animation, didUpdateTimeFrameCallback, stopCallback) {
    this.animation = animation;
    this.didUpdateTimeFrameCallback = didUpdateTimeFrameCallback;
    this.stopCallback = stopCallback;
    this.state = PLAYBACK_STATE_READY;
  }

  get duration() {
    return this.animation?.duration;
  }

  get hasBeenStopped() {
    return (this.state == PLAYBACK_STATE_STOPPED_READY_REPLAY);
  }

  get shouldRequestNextTimeFrame() {
    return (this.state == PLAYBACK_STATE_PLAYING);
  }

  get canStart() {
    return ((this.state == PLAYBACK_STATE_READY)
      || (this.state == PLAYBACK_STATE_STOPPED_READY_REPLAY));
  }

  get hasPlayedOnce() {
    return this._hasPlayedOnce;
  }

  get isPlaying() {
    return (this.state == PLAYBACK_STATE_PLAYING);
  }

  get willUpdateFrame() {
    return (this.state == PLAYBACK_STATE_STARTING
      || this.state == PLAYBACK_STATE_PLAYING);
  }

  start() {
    if (this.canStart) {
      // console.log(`start`);
      this.state = PLAYBACK_STATE_STARTING;
      this.lastEventSequencesIndices = new Array(this.animation.eventSequences.length).fill(-1);
      this.requestNextTimeFrame();
    }
  }

  stop() {
    this.state = PLAYBACK_STATE_STOPPING;
    this.lastEventSequencesIndices = null;
    this.timeFirstFrame = null;
    this._hasPlayedOnce = true;
    this.state = PLAYBACK_STATE_STOPPED_READY_REPLAY;
    this.stopCallback?.();
  }

  updateTimeFrame(time) {
    // console.log(`updateTimeFrame ${time}`);
    if (this.hasBeenStopped) {
      return;
    }

    if (this.state == PLAYBACK_STATE_STARTING) {
      this.timeFirstFrame = time;
      this.state = PLAYBACK_STATE_PLAYING;
    }

    const withinPlaybackTime = time - this.timeFirstFrame;
    this.performAnimationEvents(withinPlaybackTime);
    this.didUpdateTimeFrameCallback?.();

    // console.log(`withinPlaybackTime ${withinPlaybackTime}`);
    // console.log(`duration ${this.duration}`);
    if (withinPlaybackTime >= this.duration) {
      this.stop();
    }
    
    if (this.shouldRequestNextTimeFrame) {
      this.requestNextTimeFrame();
    }
  }

  requestNextTimeFrame() {
    window.requestAnimationFrame((time) => { this.updateTimeFrame(time) });
  }

  performAnimationEvents(withinPlaybackTime) {
    // this code assumes that the events within one event sequence
    // do not overlap in time, but they can overlap across different
    // event sequences of the animation.

    // iterate over the event sequences
    const eventSequences = this.animation?.eventSequences;
    for (let i = 0; i < eventSequences.length; i++) {
      const eventSequence = eventSequences[i];
      const lastEventSeqIndx = this.lastEventSequencesIndices[i];
      
      // update or stop last event if it was extended 
      if (lastEventSeqIndx >= 0
        && eventSequence[lastEventSeqIndx].type == EVENT_TYPE_EXTENDED) {
        let event = eventSequence[lastEventSeqIndx];
        let withinEventTime = withinPlaybackTime - event.startTime;
        if (withinEventTime > event.duration) {
          event.stop();
        } else {
          event.update(withinEventTime / event.duration);
        }
      }

      // move on to next event if its start time has passed
      let currentEventSeqIdx;
      if ((lastEventSeqIndx < (eventSequence.length - 1))
        && (eventSequence[lastEventSeqIndx+1].startTime <= withinPlaybackTime)) {
        const nextEvent = eventSequence[lastEventSeqIndx+1];
        if (nextEvent.type == EVENT_TYPE_INSTANTANEOUS) {
          nextEvent.occur();
        } else if (nextEvent.type == EVENT_TYPE_EXTENDED) {
          nextEvent.start?.();
        }
        currentEventSeqIdx = lastEventSeqIndx + 1;
      } else {
        currentEventSeqIdx = lastEventSeqIndx;
      }

      this.lastEventSequencesIndices[i] = currentEventSeqIdx;
    }
  }
}

//
// Utility
//

function shallowCopy(object) {
  return Object.assign({}, object);
}
