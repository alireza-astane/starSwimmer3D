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
    sphere { m*<0.4952467846423551,1.1440714452819527,0.1588924045909296>, 1 }        
    sphere {  m*<0.7359818893840466,1.2727815234622781,3.146447175711479>, 1 }
    sphere {  m*<3.229955178648612,1.2461054206683273,-1.0703171208602549>, 1 }
    sphere {  m*<-1.1263685752505337,3.472545389700552,-0.8150533608250409>, 1}
    sphere { m*<-3.948017167079948,-7.255282024105797,-2.415506689600708>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7359818893840466,1.2727815234622781,3.146447175711479>, <0.4952467846423551,1.1440714452819527,0.1588924045909296>, 0.5 }
    cylinder { m*<3.229955178648612,1.2461054206683273,-1.0703171208602549>, <0.4952467846423551,1.1440714452819527,0.1588924045909296>, 0.5}
    cylinder { m*<-1.1263685752505337,3.472545389700552,-0.8150533608250409>, <0.4952467846423551,1.1440714452819527,0.1588924045909296>, 0.5 }
    cylinder {  m*<-3.948017167079948,-7.255282024105797,-2.415506689600708>, <0.4952467846423551,1.1440714452819527,0.1588924045909296>, 0.5}

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
    sphere { m*<0.4952467846423551,1.1440714452819527,0.1588924045909296>, 1 }        
    sphere {  m*<0.7359818893840466,1.2727815234622781,3.146447175711479>, 1 }
    sphere {  m*<3.229955178648612,1.2461054206683273,-1.0703171208602549>, 1 }
    sphere {  m*<-1.1263685752505337,3.472545389700552,-0.8150533608250409>, 1}
    sphere { m*<-3.948017167079948,-7.255282024105797,-2.415506689600708>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7359818893840466,1.2727815234622781,3.146447175711479>, <0.4952467846423551,1.1440714452819527,0.1588924045909296>, 0.5 }
    cylinder { m*<3.229955178648612,1.2461054206683273,-1.0703171208602549>, <0.4952467846423551,1.1440714452819527,0.1588924045909296>, 0.5}
    cylinder { m*<-1.1263685752505337,3.472545389700552,-0.8150533608250409>, <0.4952467846423551,1.1440714452819527,0.1588924045909296>, 0.5 }
    cylinder {  m*<-3.948017167079948,-7.255282024105797,-2.415506689600708>, <0.4952467846423551,1.1440714452819527,0.1588924045909296>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    