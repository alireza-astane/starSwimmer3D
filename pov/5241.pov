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
    sphere { m*<-0.5566343343301307,-0.1494408773595432,-1.5561862395877324>, 1 }        
    sphere {  m*<0.41294130312787763,0.288143206417227,8.387065451218888>, 1 }
    sphere {  m*<3.68959695904153,0.005656373526449154,-3.5509758068621986>, 1 }
    sphere {  m*<-2.192857353936548,2.179207773974623,-2.5049563268562225>, 1}
    sphere { m*<-1.9250701328987163,-2.7084841684292744,-2.315410041693652>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.41294130312787763,0.288143206417227,8.387065451218888>, <-0.5566343343301307,-0.1494408773595432,-1.5561862395877324>, 0.5 }
    cylinder { m*<3.68959695904153,0.005656373526449154,-3.5509758068621986>, <-0.5566343343301307,-0.1494408773595432,-1.5561862395877324>, 0.5}
    cylinder { m*<-2.192857353936548,2.179207773974623,-2.5049563268562225>, <-0.5566343343301307,-0.1494408773595432,-1.5561862395877324>, 0.5 }
    cylinder {  m*<-1.9250701328987163,-2.7084841684292744,-2.315410041693652>, <-0.5566343343301307,-0.1494408773595432,-1.5561862395877324>, 0.5}

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
    sphere { m*<-0.5566343343301307,-0.1494408773595432,-1.5561862395877324>, 1 }        
    sphere {  m*<0.41294130312787763,0.288143206417227,8.387065451218888>, 1 }
    sphere {  m*<3.68959695904153,0.005656373526449154,-3.5509758068621986>, 1 }
    sphere {  m*<-2.192857353936548,2.179207773974623,-2.5049563268562225>, 1}
    sphere { m*<-1.9250701328987163,-2.7084841684292744,-2.315410041693652>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.41294130312787763,0.288143206417227,8.387065451218888>, <-0.5566343343301307,-0.1494408773595432,-1.5561862395877324>, 0.5 }
    cylinder { m*<3.68959695904153,0.005656373526449154,-3.5509758068621986>, <-0.5566343343301307,-0.1494408773595432,-1.5561862395877324>, 0.5}
    cylinder { m*<-2.192857353936548,2.179207773974623,-2.5049563268562225>, <-0.5566343343301307,-0.1494408773595432,-1.5561862395877324>, 0.5 }
    cylinder {  m*<-1.9250701328987163,-2.7084841684292744,-2.315410041693652>, <-0.5566343343301307,-0.1494408773595432,-1.5561862395877324>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    