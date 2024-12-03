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
    sphere { m*<-0.3379914374003937,-0.2566195875037207,-0.4060836323322763>, 1 }        
    sphere {  m*<1.0811760567997668,0.7333193263761955,9.443206464702861>, 1 }
    sphere {  m*<8.448963255122553,0.4482270755839328,-5.127470964371057>, 1 }
    sphere {  m*<-6.446999938566431,6.971308449204568,-3.6366640611894523>, 1}
    sphere { m*<-4.311782001022229,-8.910759521407968,-2.246296566153678>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0811760567997668,0.7333193263761955,9.443206464702861>, <-0.3379914374003937,-0.2566195875037207,-0.4060836323322763>, 0.5 }
    cylinder { m*<8.448963255122553,0.4482270755839328,-5.127470964371057>, <-0.3379914374003937,-0.2566195875037207,-0.4060836323322763>, 0.5}
    cylinder { m*<-6.446999938566431,6.971308449204568,-3.6366640611894523>, <-0.3379914374003937,-0.2566195875037207,-0.4060836323322763>, 0.5 }
    cylinder {  m*<-4.311782001022229,-8.910759521407968,-2.246296566153678>, <-0.3379914374003937,-0.2566195875037207,-0.4060836323322763>, 0.5}

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
    sphere { m*<-0.3379914374003937,-0.2566195875037207,-0.4060836323322763>, 1 }        
    sphere {  m*<1.0811760567997668,0.7333193263761955,9.443206464702861>, 1 }
    sphere {  m*<8.448963255122553,0.4482270755839328,-5.127470964371057>, 1 }
    sphere {  m*<-6.446999938566431,6.971308449204568,-3.6366640611894523>, 1}
    sphere { m*<-4.311782001022229,-8.910759521407968,-2.246296566153678>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0811760567997668,0.7333193263761955,9.443206464702861>, <-0.3379914374003937,-0.2566195875037207,-0.4060836323322763>, 0.5 }
    cylinder { m*<8.448963255122553,0.4482270755839328,-5.127470964371057>, <-0.3379914374003937,-0.2566195875037207,-0.4060836323322763>, 0.5}
    cylinder { m*<-6.446999938566431,6.971308449204568,-3.6366640611894523>, <-0.3379914374003937,-0.2566195875037207,-0.4060836323322763>, 0.5 }
    cylinder {  m*<-4.311782001022229,-8.910759521407968,-2.246296566153678>, <-0.3379914374003937,-0.2566195875037207,-0.4060836323322763>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    