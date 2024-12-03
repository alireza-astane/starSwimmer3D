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
    sphere { m*<1.107418046100995,-6.873048879787771e-19,0.7217747415337846>, 1 }        
    sphere {  m*<1.3060661952505588,9.662524731647062e-19,3.715198658068304>, 1 }
    sphere {  m*<4.86651674886935,5.798847763277731e-18,-0.8949762908661858>, 1 }
    sphere {  m*<-3.8170140092819715,8.164965809277259,-2.2921063805405417>, 1}
    sphere { m*<-3.8170140092819715,-8.164965809277259,-2.2921063805405444>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3060661952505588,9.662524731647062e-19,3.715198658068304>, <1.107418046100995,-6.873048879787771e-19,0.7217747415337846>, 0.5 }
    cylinder { m*<4.86651674886935,5.798847763277731e-18,-0.8949762908661858>, <1.107418046100995,-6.873048879787771e-19,0.7217747415337846>, 0.5}
    cylinder { m*<-3.8170140092819715,8.164965809277259,-2.2921063805405417>, <1.107418046100995,-6.873048879787771e-19,0.7217747415337846>, 0.5 }
    cylinder {  m*<-3.8170140092819715,-8.164965809277259,-2.2921063805405444>, <1.107418046100995,-6.873048879787771e-19,0.7217747415337846>, 0.5}

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
    sphere { m*<1.107418046100995,-6.873048879787771e-19,0.7217747415337846>, 1 }        
    sphere {  m*<1.3060661952505588,9.662524731647062e-19,3.715198658068304>, 1 }
    sphere {  m*<4.86651674886935,5.798847763277731e-18,-0.8949762908661858>, 1 }
    sphere {  m*<-3.8170140092819715,8.164965809277259,-2.2921063805405417>, 1}
    sphere { m*<-3.8170140092819715,-8.164965809277259,-2.2921063805405444>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3060661952505588,9.662524731647062e-19,3.715198658068304>, <1.107418046100995,-6.873048879787771e-19,0.7217747415337846>, 0.5 }
    cylinder { m*<4.86651674886935,5.798847763277731e-18,-0.8949762908661858>, <1.107418046100995,-6.873048879787771e-19,0.7217747415337846>, 0.5}
    cylinder { m*<-3.8170140092819715,8.164965809277259,-2.2921063805405417>, <1.107418046100995,-6.873048879787771e-19,0.7217747415337846>, 0.5 }
    cylinder {  m*<-3.8170140092819715,-8.164965809277259,-2.2921063805405444>, <1.107418046100995,-6.873048879787771e-19,0.7217747415337846>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    