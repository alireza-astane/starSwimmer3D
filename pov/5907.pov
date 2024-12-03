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
    sphere { m*<-1.462052629574789,-0.18129292602241537,-1.0993172401635865>, 1 }        
    sphere {  m*<-0.048475163064767735,0.27847211403036604,8.789545262087522>, 1 }
    sphere {  m*<6.816124958983807,0.10322320452937639,-5.449807119796799>, 1 }
    sphere {  m*<-3.135128570338259,2.1479801574835107,-1.9798237306490083>, 1}
    sphere { m*<-2.867341349300428,-2.7397117849203867,-1.7902774454864376>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.048475163064767735,0.27847211403036604,8.789545262087522>, <-1.462052629574789,-0.18129292602241537,-1.0993172401635865>, 0.5 }
    cylinder { m*<6.816124958983807,0.10322320452937639,-5.449807119796799>, <-1.462052629574789,-0.18129292602241537,-1.0993172401635865>, 0.5}
    cylinder { m*<-3.135128570338259,2.1479801574835107,-1.9798237306490083>, <-1.462052629574789,-0.18129292602241537,-1.0993172401635865>, 0.5 }
    cylinder {  m*<-2.867341349300428,-2.7397117849203867,-1.7902774454864376>, <-1.462052629574789,-0.18129292602241537,-1.0993172401635865>, 0.5}

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
    sphere { m*<-1.462052629574789,-0.18129292602241537,-1.0993172401635865>, 1 }        
    sphere {  m*<-0.048475163064767735,0.27847211403036604,8.789545262087522>, 1 }
    sphere {  m*<6.816124958983807,0.10322320452937639,-5.449807119796799>, 1 }
    sphere {  m*<-3.135128570338259,2.1479801574835107,-1.9798237306490083>, 1}
    sphere { m*<-2.867341349300428,-2.7397117849203867,-1.7902774454864376>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.048475163064767735,0.27847211403036604,8.789545262087522>, <-1.462052629574789,-0.18129292602241537,-1.0993172401635865>, 0.5 }
    cylinder { m*<6.816124958983807,0.10322320452937639,-5.449807119796799>, <-1.462052629574789,-0.18129292602241537,-1.0993172401635865>, 0.5}
    cylinder { m*<-3.135128570338259,2.1479801574835107,-1.9798237306490083>, <-1.462052629574789,-0.18129292602241537,-1.0993172401635865>, 0.5 }
    cylinder {  m*<-2.867341349300428,-2.7397117849203867,-1.7902774454864376>, <-1.462052629574789,-0.18129292602241537,-1.0993172401635865>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    