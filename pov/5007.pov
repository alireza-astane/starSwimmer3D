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
    sphere { m*<-0.2761849528243108,-0.13854989591021052,-1.6711977555998259>, 1 }        
    sphere {  m*<0.5314200448612743,0.29074905691842196,8.286889654300888>, 1 }
    sphere {  m*<2.5089276474525968,-0.03470214950748386,-2.924791632849869>, 1 }
    sphere {  m*<-1.8982568344998312,2.189928946976999,-2.644371243952447>, 1}
    sphere { m*<-1.6304696134619994,-2.6977629954268982,-2.4548249587898767>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5314200448612743,0.29074905691842196,8.286889654300888>, <-0.2761849528243108,-0.13854989591021052,-1.6711977555998259>, 0.5 }
    cylinder { m*<2.5089276474525968,-0.03470214950748386,-2.924791632849869>, <-0.2761849528243108,-0.13854989591021052,-1.6711977555998259>, 0.5}
    cylinder { m*<-1.8982568344998312,2.189928946976999,-2.644371243952447>, <-0.2761849528243108,-0.13854989591021052,-1.6711977555998259>, 0.5 }
    cylinder {  m*<-1.6304696134619994,-2.6977629954268982,-2.4548249587898767>, <-0.2761849528243108,-0.13854989591021052,-1.6711977555998259>, 0.5}

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
    sphere { m*<-0.2761849528243108,-0.13854989591021052,-1.6711977555998259>, 1 }        
    sphere {  m*<0.5314200448612743,0.29074905691842196,8.286889654300888>, 1 }
    sphere {  m*<2.5089276474525968,-0.03470214950748386,-2.924791632849869>, 1 }
    sphere {  m*<-1.8982568344998312,2.189928946976999,-2.644371243952447>, 1}
    sphere { m*<-1.6304696134619994,-2.6977629954268982,-2.4548249587898767>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5314200448612743,0.29074905691842196,8.286889654300888>, <-0.2761849528243108,-0.13854989591021052,-1.6711977555998259>, 0.5 }
    cylinder { m*<2.5089276474525968,-0.03470214950748386,-2.924791632849869>, <-0.2761849528243108,-0.13854989591021052,-1.6711977555998259>, 0.5}
    cylinder { m*<-1.8982568344998312,2.189928946976999,-2.644371243952447>, <-0.2761849528243108,-0.13854989591021052,-1.6711977555998259>, 0.5 }
    cylinder {  m*<-1.6304696134619994,-2.6977629954268982,-2.4548249587898767>, <-0.2761849528243108,-0.13854989591021052,-1.6711977555998259>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    