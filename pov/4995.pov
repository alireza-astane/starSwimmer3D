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
    sphere { m*<-0.26689171983750365,-0.13791262611913335,-1.667862738609501>, 1 }        
    sphere {  m*<0.5333117683238784,0.28991967375226735,8.262769320595202>, 1 }
    sphere {  m*<2.4678166741687533,-0.03587865073275917,-2.8970722640606827>, 1 }
    sphere {  m*<-1.8885070797303938,2.1905613182994657,-2.641808504025469>, 1}
    sphere { m*<-1.620719858692562,-2.6971306241044317,-2.4522622188628964>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5333117683238784,0.28991967375226735,8.262769320595202>, <-0.26689171983750365,-0.13791262611913335,-1.667862738609501>, 0.5 }
    cylinder { m*<2.4678166741687533,-0.03587865073275917,-2.8970722640606827>, <-0.26689171983750365,-0.13791262611913335,-1.667862738609501>, 0.5}
    cylinder { m*<-1.8885070797303938,2.1905613182994657,-2.641808504025469>, <-0.26689171983750365,-0.13791262611913335,-1.667862738609501>, 0.5 }
    cylinder {  m*<-1.620719858692562,-2.6971306241044317,-2.4522622188628964>, <-0.26689171983750365,-0.13791262611913335,-1.667862738609501>, 0.5}

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
    sphere { m*<-0.26689171983750365,-0.13791262611913335,-1.667862738609501>, 1 }        
    sphere {  m*<0.5333117683238784,0.28991967375226735,8.262769320595202>, 1 }
    sphere {  m*<2.4678166741687533,-0.03587865073275917,-2.8970722640606827>, 1 }
    sphere {  m*<-1.8885070797303938,2.1905613182994657,-2.641808504025469>, 1}
    sphere { m*<-1.620719858692562,-2.6971306241044317,-2.4522622188628964>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5333117683238784,0.28991967375226735,8.262769320595202>, <-0.26689171983750365,-0.13791262611913335,-1.667862738609501>, 0.5 }
    cylinder { m*<2.4678166741687533,-0.03587865073275917,-2.8970722640606827>, <-0.26689171983750365,-0.13791262611913335,-1.667862738609501>, 0.5}
    cylinder { m*<-1.8885070797303938,2.1905613182994657,-2.641808504025469>, <-0.26689171983750365,-0.13791262611913335,-1.667862738609501>, 0.5 }
    cylinder {  m*<-1.620719858692562,-2.6971306241044317,-2.4522622188628964>, <-0.26689171983750365,-0.13791262611913335,-1.667862738609501>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    