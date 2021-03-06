(ros::roseus "fetch")

(require "package://fetcheus/fetch-utils.l")
(require "package://pr2eus/robot-interface.l")
(require "package://pr2eus_moveit/euslisp/robot-moveit.l")

(ros::load-ros-manifest "fetcheus")

(defclass fetch-interface
  :super robot-move-base-interface
  :slots (gripper-action moveit-robot)
  )

(defmethod fetch-interface
  (:init (&rest args)
   (prog1 (send-super* :init :robot fetch-robot :base-frame-id "/base_link" :odom-topic "/odom_combined" :base-controller-action-name nil args)
     (send self :add-controller :arm-controller)
     (send self :add-controller :torso-controller)
     (send self :add-controller :head-controller)
     (ros::subscribe "gripper_state" fetch_driver_msgs::GripperState #'send self :fetch-gripper-state-callback :groupname groupname)
     (setq gripper-action
           (instance ros::simple-action-client :init
                     "/gripper_controller/gripper_action"
                     control_msgs::GripperCommandAction
                     :groupname groupname))
     (setq moveit-robot (instance fetch-robot :init))
     (send self :set-moveit-environment (instance fetch-moveit-environment :init :robot moveit-robot))
     ))
  (:state (&rest args)
   ":state calls with :wait-until-update by default, since Fetch publishes /joint_states from body and gripper at almost same frequency"
   (send-super* :state (if (member :wait-until-update args) args (append args (list :wait-until-update t)))))
  (:check-continuous-joint-move-over-180 ;; can be removed if http//github.com/jsk-ros-pkg/jsk_pr2eus/pull/322 merged
   (diff-av)
   (let ((i 0) add-new-trajectory-point)
     (dolist (j (send robot :joint-list))
       ;; for continuous rotational joint
       (when (and (> (- (send j :max-angle) (send j :min-angle)) 360)
                  (> (abs (elt diff-av i)) 180))
         (ros::ros-warn "continuous joint (~A) moves ~A degree, commanded joint differs from original trajectory to avoid unintentional 360 rotation" (send j :name) (elt diff-av i))
         (setq add-new-trajectory-point t))
       (incf i (send j :joint-dof)))
     add-new-trajectory-point))
  (:angle-vector-raw (av &optional (tm 3000) &rest args)
   (let* ((prev-av (send self :state :potentio-vector :wait-until-update t))
          (diff-av (v- av prev-av)))
     (when (send self :check-continuous-joint-move-over-180 diff-av)
       (let* (avs (minjerk (instance minjerk-interpolator :init))
                  (scale-av (send self :sub-angle-vector av prev-av))
                  dist div)
         (setq dist (abs (geo::find-extream (coerce diff-av cons) #'abs #'>=)))
         (setq div (round (/ dist 120.0)))
         (send minjerk :reset
               :position-list (list prev-av (v+ prev-av scale-av))
               :time-list (list tm))
         (send minjerk :start-interpolation)
         (send minjerk :pass-time (/ tm div))
         (dotimes (i div)
           (setq avs (append avs (list (send minjerk :pass-time (/ tm div))))))
         (send* self :angle-vector-sequence-raw avs (make-list div :initial-element (/ tm div)) args)
         (return-from :angle-vector-raw (car (last avs)))))
     (send-super* :angle-vector av tm args)))
  (:angle-vector-sequence-raw (&rest args) (send-super* :angle-vector-sequence args))
  (:angle-vector
   (av &optional (tm 3000) &rest args) ;; (ctype controller-type) (start-time 0) &rest args
                                       ;;  &key (use-torso t) (clear-velocities t) &allow-other-keys)
   "Send joind angle to robot with self-collision motion planning, this method retuns immediately, so use :wait-interpolation to block until the motion stops.
- av : joint angle vector [rad]
- tm : time to goal in [msec]
- use-torso : set t to use torso
"
   (let ((ctype controller-type) (start-time 0) (use-torso t) (clear-velocities t))
   ;; as of 0.3.x, :angle-vector (robot-interface) :acceps tm ctype start-time as optional arguments, but in here we prefer old API
   (if (= (length args) 1) ;; args must be ctype
       (setq ctype (car args)
             args (cdr args)))
   (if (and (>= (length args) 2) (null (member (car args) '(:use-torso :start-time :clear-velocities))));; args must be ctype start-time
       (setq ctype (car args)
             start-time (cadr args)
             args (cddr args)))
   (if (member :use-torso args) (setq use-torso (cadr (member :use-torso args))))
   (if (member :start-time args) (setq use-torso (cadr (member :start-time args))))
   (if (member :clear-velocities args) (setq clear-velocities (cadr (member :clear-velocities args))))
   ;; for simulation mode
   (when (send self :simulation-modep)
     (return-from :angle-vector (send* self :angle-vector-raw av tm ctype start-time args)))
   ;;
   (when (not (numberp tm))
     (ros::warn ":angle-vector tm is not a number, use :angle-vector av tm args"))
   (send* self :angle-vector-motion-plan av :ctype ctype :move-arm :rarm :total-time tm
               :start-offset-time start-time :clear-velocities clear-velocities
               :use-torso use-torso args)))
  (:angle-vector-sequence
   (avs &optional tms &rest args) ;; (ctype controller-type) (start-time 0) &rest args
                                  ;; &key (use-torso t) (clear-velocities t) &allow-other-keys)
   "Send joind angle to robot with self-collision motion planning, this method returns immediately, so use :wait-interpolation to block until the motion stops.
- avs : sequence of joint angle vector [rad]
- tms : list of time to goal from previous angle-vector point in [msec]
- use-torso : set t to use torso
"
   (let ((ctype controller-type) (start-time 0) (use-torso t) (clear-velocities t))
   ;; as of 0.3.x, :angle-vector (robot-interface) :acceps tm ctype start-time as optional arguments, but in here we prefer old API
   (if (= (length args) 1) ;; args must be ctype
       (setq ctype (car args)
             args (cdr args)))
   (if (and (>= (length args) 2) (null (member (car args) '(:use-torso :start-time :clear-velocities))));; args must be ctype start-time
       (setq ctype (car args)
             start-time (cadr args)
             args (cddr args)))
   (if (member :use-torso args) (setq use-torso (cadr (member :use-torso args))))
   (if (member :start-time args) (setq use-torso (cadr (member :start-time args))))
   (if (member :clear-velocities args) (setq clear-velocities (cadr (member :clear-velocities args))))
   ;; for simulation mode
   (when (send self :simulation-modep)
     (return-from :angle-vector-sequence
                  (send* self :angle-vector-sequence-raw avs tms ctype start-time args)))
   (unless (and (listp tms) (every #'numberp tms))
     (ros::warn ":angle-vector-sequence tms is not a list of number, use :angle-vector-sequence av tms args"))
   (if tms
     (setq tms (apply #'+ tms))
     (setq tms 3000))
   (send* self :angle-vector-motion-plan avs :ctype ctype :move-arm :rarm :total-time tms
               :start-offset-time start-time :clear-velocities clear-velocities
               :use-torso use-torso args)))
  (:default-controller ()
   (append
    (send self :arm-controller)
    (send self :torso-controller)
    (send self :head-controller)))
  (:arm-controller ()
   (list
    (list
     (cons :controller-action "/arm_controller/follow_joint_trajectory")
     (cons :controller-state "/arm_controller/follow_joint_trajectory/state")
     (cons :action-type control_msgs::FollowJointTrajectoryAction)
     (cons :joint-names (list "shoulder_pan_joint" "shoulder_lift_joint" "upperarm_roll_joint" "elbow_flex_joint" "forearm_roll_joint" "wrist_flex_joint" "wrist_roll_joint")))))
  (:torso-controller ()
   (list
    (list
     (cons :controller-action "/torso_controller/follow_joint_trajectory")
     (cons :controller-state "/torso_controller/follow_joint_trajectory/state")
     (cons :action-type control_msgs::FollowJointTrajectoryAction)
     (cons :joint-names (list "torso_lift_joint")))))
  (:head-controller ()
   (list
    (list
     (cons :controller-action "/head_controller/follow_joint_trajectory")
     (cons :controller-state "/head_controller/follow_joint_trajectory/state")
     (cons :action-type control_msgs::FollowJointTrajectoryAction)
     (cons :joint-names (list "head_pan_joint" "head_tilt_joint")))))
  (:start-grasp
    (&key (effort) (wait t))
    (send self :go-grasp :pos 0 :effort effort :wait wait))
  (:stop-grasp
    (&key (effort) (wait t))
    (send self :go-grasp :pos 0.1 :effort effort :wait wait))
  (:go-grasp
    (&key (pos 0) (effort) (wait t))
    (when (send self :simulation-modep)
      (send robot :l_gripper_finger_joint :joint-angle (/ (* pos 1000) 2)) ;; m -> mm
      (send robot :r_gripper_finger_joint :joint-angle (/ (* pos 1000) 2))
      (return-from :go-grasp t))
    (setq effort (or effort 40))
    (let (goal result)
      (setq goal (instance control_msgs::GripperCommandActionGoal :init))
      (send goal :goal :command :position pos)
      (send goal :goal :command :max_effort effort)
      (send gripper-action :send-goal goal)
      (when wait (send gripper-action :wait-for-result))
      (setq result (send gripper-action :get-result))
      result))
  (:fetch-gripper-state-callback
    (msg)
    (let ((msg (car (send msg :joints))))
      (dolist (slot '((position . 2) (velocity . 2) (effort . 1)))
	(send self :set-robot-state1 (intern (format nil "GRIPPER-~A" (string (car slot))) *keyword-package*) (* (cdr slot) (send msg (intern (string (car slot)) *keyword-package*)))))))
  (:gripper (key)
   "get information about gripper
Arguments:
 - key (:position :velocity :effort)
Example: (send self :gripper :position) => 0.00"
   (send-super :state (intern (format nil "GRIPPER-~A" (string key)) *keyword-package*)))
  ;;
  (:speak (text &rest args)
    (let ()
      (send-super* :speak text :topic-name "/sound_play" args)))
  )

;; interface for simple base actions
(defmethod fetch-interface
  (:send-cmd-vel-raw (x d &key (topic-name "/base_controller/command"))
    (when (send self :simulation-modep)
      (return-from :send-cmd-vel-raw t))
    (unless (ros::get-topic-publisher topic-name)
      (ros::advertise topic-name geometry_msgs::Twist 1)
      (unix:sleep 1))
    (let ((msg (instance geometry_msgs::Twist :init)))
      (send msg :linear :x x)
      (send msg :angular :z d)
      (ros::publish topic-name msg)))
  (:go-velocity
    (x y d ;; [m/sec] [m/sec] [rad/sec]
       &optional (msec 1000) ;; msec is total animation time [msec]
       &key (stop t) (wait t))
    (unless (not (zerop y))
      (ros::ros-error
        "Setting velocity in y-axis is not valid for non-holonomic robot")
      (return-from :go-velocity nil))
    (unless wait
      (ros::ros-error ":go-velocity without wait is unsupported")
      (return-from :go-velocity nil))
    (setq msec (- msec (ros::get-param "/base_controller/timeout" 0.25)))
    (ros::rate 100)
    (let ((start-time (ros::time-now)))
      (while (and (ros::ok)
                  (< (* 1000.0 (send (ros::time- (ros::time-now) start-time) :to-sec)) msec))
        (send self :spin-once)
        (send self :send-cmd-vel-raw x d)
        (ros::sleep)))
    (when stop
      (send self :send-cmd-vel-raw 0 0))
    (ros::rate 10)
    t)
  (:go-pos-unsafe
    (&rest args)
    (send self :put :go-pos-unsafe-no-wait-goal (coerce args float-vector))
    (send self :go-pos-unsafe-wait)
    t)
  (:go-pos-unsafe-no-wait
    (x y &optional (d 0)) ;; [m] [m] [degree]
    (ros::ros-warn ":go-pos-unsafe-no-wait is not supported for this robot.")
    (send self :put :go-pos-unsafe-no-wait-goal (float-vector x y d))
    t)
  (:go-pos-unsafe-wait
    (&key (translation-threshold 0.05) (rotation-threshold (deg2rad 5))
          (translation-gain 1.0) (rotation-gain 1.0))
    (unless (send self :get :go-pos-unsafe-no-wait-goal)
      (ros::ros-error ":go-pos-unsafe-wait is called without goal")
      (return-from :go-pos-unsafe-wait nil))
    (ros::rate 100)
    (labels ((shortest-angle
              (d0 d1)
              (atan2 (sin (- d0 d1)) (cos (- d0 d1)))))
      (let* ((goal (send self :get :go-pos-unsafe-no-wait-goal))
             (x (aref goal 0))
             (y (aref goal 1))
             (d (aref goal 2))
             (forward-p (>= x 0))
             (d0 (if forward-p (atan2 y x) (atan2 (- y) (- x))))
             (d1 (distance (float-vector 0 0) (float-vector x y)))
             (d2 (shortest-angle (- d d0) 0))
             org-cds cur-cds diffrot diffpos err)
        (send self :remprop :go-pos-unsafe-no-wait-goal)

        ;; yaw-x-yaw approach
        (ros::ros-info "go-pos-unsafe: (yaw, x, yaw) = (~A ~A ~A)" d0 d1 d2)

        ;; yaw
        (send *ri* :spin-once)
        (setq org-cds (send self :state :odom :pose))
        (ros::ros-debug "org-cds: ~A" org-cds)
        (while (ros::ok)
          (send *ri* :spin-once)
          (setq cur-cds (send *ri* :state :odom :pose))
          (setq diffrot (aref (send org-cds :difference-rotation cur-cds) 2))
          (setq err (shortest-angle d0 diffrot))
          (ros::ros-debug "diffrot: ~A, err: ~A" diffrot err)
          (when (< (abs err) rotation-threshold)
            (unless (send self :simulation-modep)
              (send *ri* :send-cmd-vel-raw 0 0)) ;; stop
            (return))
          (when (send self :simulation-modep)
            (send *ri* :robot :rotate (deg2rad (if (> err 0) 5 -5)) :z)
            (send *ri* :draw-objects))
          (send *ri* :send-cmd-vel-raw 0 (* rotation-gain err))
          (ros::sleep))

        ;; x
        (setq org-cds (send *ri* :state :odom :pose))
        (while (ros::ok)
          (send *ri* :spin-once)
          (setq cur-cds (send *ri* :state :odom :pose))
          (setq diffpos (* 0.001 (norm (send org-cds :difference-position cur-cds))))
          (setq err (- d1 diffpos))
          (ros::ros-debug "diffpos: ~A, err: ~A" diffpos err)
          (when (< (abs err) translation-threshold)
            (send *ri* :send-cmd-vel-raw 0 0) ;; stop
            (return))
          (when (send self :simulation-modep)
            (send *ri* :robot :newcoords
                  (send (send *ri* :robot :copy-worldcoords)
                        :translate (float-vector
                                    (if forward-p 0.1 -0.1) 0 0)))
            (send *ri* :draw-objects))
          (send *ri* :send-cmd-vel-raw
                (* translation-gain (if forward-p err (- err))) 0)
          (ros::sleep))

        ;; yaw
        (setq org-cds (send *ri* :state :odom :pose))
        (while (ros::ok)
          (send *ri* :spin-once)
          (setq cur-cds (send *ri* :state :odom :pose))
          (setq diffrot (aref (send org-cds :difference-rotation cur-cds) 2))
          (setq err (shortest-angle d2 diffrot))
          (ros::ros-debug "diffrot: ~A, err: ~A" diffrot err)
          (when (< (abs err) rotation-threshold)
            (send *ri* :send-cmd-vel-raw 0 0) ;; stop
            (return))
          (when (send self :simulation-modep)
            (send *ri* :robot :rotate (deg2rad (if (> err 0) 5 -5)) :z)
            (send *ri* :draw-objects))
          (send *ri* :send-cmd-vel-raw 0 (* rotation-gain err))
          (ros::sleep))
        t)))
) ;; fetch-interface (simple base actions)

(defun fetch-init (&optional (create-viewer))
  (unless (boundp '*fetch*) (fetch) (send *fetch* :reset-pose))
  (unless (ros::ok) (ros::roseus "fetch_eus_interface"))
  (unless (boundp '*ri*) (setq *ri* (instance fetch-interface :init)))

  (ros::spin-once)
  (send *ri* :spin-once)

  (send *fetch* :angle-vector (send *ri* :state :potentio-vector))
  (when create-viewer (objects (list *fetch*)))
  )

(defclass fetch-moveit-environment
  :super moveit-environment)
(defmethod fetch-moveit-environment
  (:init (&key ((:robot rb) *fetch*) &rest args)
         (send-super* :init :robot rb :frame-id "base_link" args))
  (:default-configuration ()
   (list (list :rarm
               (cons :group-name "arm")
               (cons :target-link
                     (send self :search-link-from-name "wrist_roll_link"))
               (cons :joint-list (send robot :rarm :joint-list))
               )
         (list :rarm-torso
               (cons :group-name "arm_with_torso")
               (cons :target-link
                     (send self :search-link-from-name "wrist_roll_link"))
               (cons :joint-list (append
                                  (send robot :torso :joint-list)
                                  (send robot :rarm :joint-list)))
               )
#|
    <group name="arm">
        <joint name="shoulder_pan_joint" />
        <joint name="shoulder_lift_joint" />
        <joint name="upperarm_roll_joint" />
        <joint name="elbow_flex_joint" />
        <joint name="forearm_roll_joint" />
        <joint name="wrist_flex_joint" />
        <joint name="wrist_roll_joint" />
    </group>
    <group name="arm_with_torso">
        <joint name="torso_lift_joint" />
        <joint name="shoulder_pan_joint" />
        <joint name="shoulder_lift_joint" />
        <joint name="upperarm_roll_joint" />
        <joint name="elbow_flex_joint" />
        <joint name="forearm_roll_joint" />
        <joint name="wrist_flex_joint" />
        <joint name="wrist_roll_joint" />
    </group>
    <!--END EFFECTOR: Purpose: Represent information about an end effector.-->
    <end_effector name="gripper" parent_link="wrist_roll_link" group="gripper" />
|#
         ))
  )

#|
(setq *ri* (instance fetch-interface :init))
(setq *fetch* (fetch))
(send *fetch* :reset-pose)
(send *ri* :angle-vector (send *fetch* :angle-vector) 2000)
|#
