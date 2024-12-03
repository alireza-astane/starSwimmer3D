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
    sphere { m*<-0.6901423061752711,-0.15443920127850339,-1.4966171411951241>, 1 }        
    sphere {  m*<0.3532060087274374,0.2868843467294318,8.438994077306472>, 1 }
    sphere {  m*<4.191205422564034,0.022143246419834783,-3.8340627427099534>, 1 }
    sphere {  m*<-2.3326859248859817,2.1742970244611044,-2.4341841746986934>, 1}
    sphere { m*<-2.0648987038481494,-2.713394917942793,-2.244637889536123>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3532060087274374,0.2868843467294318,8.438994077306472>, <-0.6901423061752711,-0.15443920127850339,-1.4966171411951241>, 0.5 }
    cylinder { m*<4.191205422564034,0.022143246419834783,-3.8340627427099534>, <-0.6901423061752711,-0.15443920127850339,-1.4966171411951241>, 0.5}
    cylinder { m*<-2.3326859248859817,2.1742970244611044,-2.4341841746986934>, <-0.6901423061752711,-0.15443920127850339,-1.4966171411951241>, 0.5 }
    cylinder {  m*<-2.0648987038481494,-2.713394917942793,-2.244637889536123>, <-0.6901423061752711,-0.15443920127850339,-1.4966171411951241>, 0.5}

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
    sphere { m*<-0.6901423061752711,-0.15443920127850339,-1.4966171411951241>, 1 }        
    sphere {  m*<0.3532060087274374,0.2868843467294318,8.438994077306472>, 1 }
    sphere {  m*<4.191205422564034,0.022143246419834783,-3.8340627427099534>, 1 }
    sphere {  m*<-2.3326859248859817,2.1742970244611044,-2.4341841746986934>, 1}
    sphere { m*<-2.0648987038481494,-2.713394917942793,-2.244637889536123>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3532060087274374,0.2868843467294318,8.438994077306472>, <-0.6901423061752711,-0.15443920127850339,-1.4966171411951241>, 0.5 }
    cylinder { m*<4.191205422564034,0.022143246419834783,-3.8340627427099534>, <-0.6901423061752711,-0.15443920127850339,-1.4966171411951241>, 0.5}
    cylinder { m*<-2.3326859248859817,2.1742970244611044,-2.4341841746986934>, <-0.6901423061752711,-0.15443920127850339,-1.4966171411951241>, 0.5 }
    cylinder {  m*<-2.0648987038481494,-2.713394917942793,-2.244637889536123>, <-0.6901423061752711,-0.15443920127850339,-1.4966171411951241>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    