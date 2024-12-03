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
    sphere { m*<-0.338184680549705,-0.14100239779792462,-1.6469302717908842>, 1 }        
    sphere {  m*<0.5058300819501342,0.29016795093324993,8.30805389912345>, 1 }
    sphere {  m*<2.792656069429025,-0.02479553090242209,-3.069892693379825>, 1 }
    sphere {  m*<-1.963470373874249,2.187511932461228,-2.6146411611719476>, 1}
    sphere { m*<-1.6956831528364171,-2.7001800099426694,-2.425094876009377>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5058300819501342,0.29016795093324993,8.30805389912345>, <-0.338184680549705,-0.14100239779792462,-1.6469302717908842>, 0.5 }
    cylinder { m*<2.792656069429025,-0.02479553090242209,-3.069892693379825>, <-0.338184680549705,-0.14100239779792462,-1.6469302717908842>, 0.5}
    cylinder { m*<-1.963470373874249,2.187511932461228,-2.6146411611719476>, <-0.338184680549705,-0.14100239779792462,-1.6469302717908842>, 0.5 }
    cylinder {  m*<-1.6956831528364171,-2.7001800099426694,-2.425094876009377>, <-0.338184680549705,-0.14100239779792462,-1.6469302717908842>, 0.5}

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
    sphere { m*<-0.338184680549705,-0.14100239779792462,-1.6469302717908842>, 1 }        
    sphere {  m*<0.5058300819501342,0.29016795093324993,8.30805389912345>, 1 }
    sphere {  m*<2.792656069429025,-0.02479553090242209,-3.069892693379825>, 1 }
    sphere {  m*<-1.963470373874249,2.187511932461228,-2.6146411611719476>, 1}
    sphere { m*<-1.6956831528364171,-2.7001800099426694,-2.425094876009377>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5058300819501342,0.29016795093324993,8.30805389912345>, <-0.338184680549705,-0.14100239779792462,-1.6469302717908842>, 0.5 }
    cylinder { m*<2.792656069429025,-0.02479553090242209,-3.069892693379825>, <-0.338184680549705,-0.14100239779792462,-1.6469302717908842>, 0.5}
    cylinder { m*<-1.963470373874249,2.187511932461228,-2.6146411611719476>, <-0.338184680549705,-0.14100239779792462,-1.6469302717908842>, 0.5 }
    cylinder {  m*<-1.6956831528364171,-2.7001800099426694,-2.425094876009377>, <-0.338184680549705,-0.14100239779792462,-1.6469302717908842>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    