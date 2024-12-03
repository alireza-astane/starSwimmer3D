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
    sphere { m*<-1.4243899812127678,-0.4634757730973592,-0.9397968144671477>, 1 }        
    sphere {  m*<0.032156196221030164,0.10930473906635985,8.937047735205969>, 1 }
    sphere {  m*<7.387507634221003,0.0203844630720027,-5.642445554839384>, 1 }
    sphere {  m*<-4.203100063894444,3.1900479262613946,-2.364042894064381>, 1}
    sphere { m*<-2.7703036128110083,-3.060981524841464,-1.6029662433579595>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.032156196221030164,0.10930473906635985,8.937047735205969>, <-1.4243899812127678,-0.4634757730973592,-0.9397968144671477>, 0.5 }
    cylinder { m*<7.387507634221003,0.0203844630720027,-5.642445554839384>, <-1.4243899812127678,-0.4634757730973592,-0.9397968144671477>, 0.5}
    cylinder { m*<-4.203100063894444,3.1900479262613946,-2.364042894064381>, <-1.4243899812127678,-0.4634757730973592,-0.9397968144671477>, 0.5 }
    cylinder {  m*<-2.7703036128110083,-3.060981524841464,-1.6029662433579595>, <-1.4243899812127678,-0.4634757730973592,-0.9397968144671477>, 0.5}

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
    sphere { m*<-1.4243899812127678,-0.4634757730973592,-0.9397968144671477>, 1 }        
    sphere {  m*<0.032156196221030164,0.10930473906635985,8.937047735205969>, 1 }
    sphere {  m*<7.387507634221003,0.0203844630720027,-5.642445554839384>, 1 }
    sphere {  m*<-4.203100063894444,3.1900479262613946,-2.364042894064381>, 1}
    sphere { m*<-2.7703036128110083,-3.060981524841464,-1.6029662433579595>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.032156196221030164,0.10930473906635985,8.937047735205969>, <-1.4243899812127678,-0.4634757730973592,-0.9397968144671477>, 0.5 }
    cylinder { m*<7.387507634221003,0.0203844630720027,-5.642445554839384>, <-1.4243899812127678,-0.4634757730973592,-0.9397968144671477>, 0.5}
    cylinder { m*<-4.203100063894444,3.1900479262613946,-2.364042894064381>, <-1.4243899812127678,-0.4634757730973592,-0.9397968144671477>, 0.5 }
    cylinder {  m*<-2.7703036128110083,-3.060981524841464,-1.6029662433579595>, <-1.4243899812127678,-0.4634757730973592,-0.9397968144671477>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    