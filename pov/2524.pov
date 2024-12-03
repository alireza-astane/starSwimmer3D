#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.8703225407758058,0.6750203429506194,0.3804602314654797>, 1 }        
    sphere {  m*<1.1137388927643421,0.7328545412805735,3.370006949740798>, 1 }
    sphere {  m*<3.6069860818268777,0.7328545412805733,-0.8472752587498202>, 1 }
    sphere {  m*<-2.4198776999719267,5.714438419984646,-1.5649118420534707>, 1}
    sphere { m*<-3.861584652518666,-7.678172619493353,-2.416678074279292>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1137388927643421,0.7328545412805735,3.370006949740798>, <0.8703225407758058,0.6750203429506194,0.3804602314654797>, 0.5 }
    cylinder { m*<3.6069860818268777,0.7328545412805733,-0.8472752587498202>, <0.8703225407758058,0.6750203429506194,0.3804602314654797>, 0.5}
    cylinder { m*<-2.4198776999719267,5.714438419984646,-1.5649118420534707>, <0.8703225407758058,0.6750203429506194,0.3804602314654797>, 0.5 }
    cylinder {  m*<-3.861584652518666,-7.678172619493353,-2.416678074279292>, <0.8703225407758058,0.6750203429506194,0.3804602314654797>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.8703225407758058,0.6750203429506194,0.3804602314654797>, 1 }        
    sphere {  m*<1.1137388927643421,0.7328545412805735,3.370006949740798>, 1 }
    sphere {  m*<3.6069860818268777,0.7328545412805733,-0.8472752587498202>, 1 }
    sphere {  m*<-2.4198776999719267,5.714438419984646,-1.5649118420534707>, 1}
    sphere { m*<-3.861584652518666,-7.678172619493353,-2.416678074279292>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1137388927643421,0.7328545412805735,3.370006949740798>, <0.8703225407758058,0.6750203429506194,0.3804602314654797>, 0.5 }
    cylinder { m*<3.6069860818268777,0.7328545412805733,-0.8472752587498202>, <0.8703225407758058,0.6750203429506194,0.3804602314654797>, 0.5}
    cylinder { m*<-2.4198776999719267,5.714438419984646,-1.5649118420534707>, <0.8703225407758058,0.6750203429506194,0.3804602314654797>, 0.5 }
    cylinder {  m*<-3.861584652518666,-7.678172619493353,-2.416678074279292>, <0.8703225407758058,0.6750203429506194,0.3804602314654797>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    