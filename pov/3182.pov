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
    sphere { m*<0.37466981919647024,0.9161379792420677,0.08903087171846989>, 1 }        
    sphere {  m*<0.6154049239381622,1.0448480574223935,3.076585642839023>, 1 }
    sphere {  m*<3.1093782132027274,1.0181719546284422,-1.1401786537327139>, 1 }
    sphere {  m*<-1.2469455406964194,3.2446119236606696,-0.8849148936974999>, 1}
    sphere { m*<-3.560006028362615,-6.521802595569202,-2.1906954843913113>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6154049239381622,1.0448480574223935,3.076585642839023>, <0.37466981919647024,0.9161379792420677,0.08903087171846989>, 0.5 }
    cylinder { m*<3.1093782132027274,1.0181719546284422,-1.1401786537327139>, <0.37466981919647024,0.9161379792420677,0.08903087171846989>, 0.5}
    cylinder { m*<-1.2469455406964194,3.2446119236606696,-0.8849148936974999>, <0.37466981919647024,0.9161379792420677,0.08903087171846989>, 0.5 }
    cylinder {  m*<-3.560006028362615,-6.521802595569202,-2.1906954843913113>, <0.37466981919647024,0.9161379792420677,0.08903087171846989>, 0.5}

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
    sphere { m*<0.37466981919647024,0.9161379792420677,0.08903087171846989>, 1 }        
    sphere {  m*<0.6154049239381622,1.0448480574223935,3.076585642839023>, 1 }
    sphere {  m*<3.1093782132027274,1.0181719546284422,-1.1401786537327139>, 1 }
    sphere {  m*<-1.2469455406964194,3.2446119236606696,-0.8849148936974999>, 1}
    sphere { m*<-3.560006028362615,-6.521802595569202,-2.1906954843913113>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6154049239381622,1.0448480574223935,3.076585642839023>, <0.37466981919647024,0.9161379792420677,0.08903087171846989>, 0.5 }
    cylinder { m*<3.1093782132027274,1.0181719546284422,-1.1401786537327139>, <0.37466981919647024,0.9161379792420677,0.08903087171846989>, 0.5}
    cylinder { m*<-1.2469455406964194,3.2446119236606696,-0.8849148936974999>, <0.37466981919647024,0.9161379792420677,0.08903087171846989>, 0.5 }
    cylinder {  m*<-3.560006028362615,-6.521802595569202,-2.1906954843913113>, <0.37466981919647024,0.9161379792420677,0.08903087171846989>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    