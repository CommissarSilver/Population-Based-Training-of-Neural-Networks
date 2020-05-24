from preprocess_frame import *

def stack_frames(stacked_frames, state, is_new_episode, stack_size=4):
    # Arguments:
    #   stacked_frames: A deque containing four frames
    #   state: Current frame
    #   is_new_episode: A boolean value which determines if we're stacking frames from a newly created episode or not
    #   stack_size: Determines the stack size. By default is 4.
    # Returns:
    #   stacked_state: a 84*84*4 deque containing the last 4 states(frames).
    # Implements:
    #   Calls the preprocess function on each new frame and then stacks #stack_size of frames together in a deque.

    # preprocess the frame
    frame = preprocess_frame(state).T
    # if we're in a new episode create a new deque and stack four of the first frames in it. if not just add it to the stack
    if is_new_episode:
        for i in range(stack_size):
            stacked_frames.append(frame)

    else:
        stacked_frames.append(frame)

    return stacked_frames
