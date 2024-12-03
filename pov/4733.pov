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
    sphere { m*<-0.2322136415213687,-0.1193718396559276,-1.2375031596115538>, 1 }        
    sphere {  m*<0.420820452607926,0.22977569908790096,6.866737082848121>, 1 }
    sphere {  m*<2.5024947524848886,-0.017337864269553388,-2.4667126850627343>, 1 }
    sphere {  m*<-1.8538290014142587,2.209102104762671,-2.211448925027521>, 1}
    sphere { m*<-1.5860417803764268,-2.6785898376412263,-2.0219026398649484>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.420820452607926,0.22977569908790096,6.866737082848121>, <-0.2322136415213687,-0.1193718396559276,-1.2375031596115538>, 0.5 }
    cylinder { m*<2.5024947524848886,-0.017337864269553388,-2.4667126850627343>, <-0.2322136415213687,-0.1193718396559276,-1.2375031596115538>, 0.5}
    cylinder { m*<-1.8538290014142587,2.209102104762671,-2.211448925027521>, <-0.2322136415213687,-0.1193718396559276,-1.2375031596115538>, 0.5 }
    cylinder {  m*<-1.5860417803764268,-2.6785898376412263,-2.0219026398649484>, <-0.2322136415213687,-0.1193718396559276,-1.2375031596115538>, 0.5}

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
    sphere { m*<-0.2322136415213687,-0.1193718396559276,-1.2375031596115538>, 1 }        
    sphere {  m*<0.420820452607926,0.22977569908790096,6.866737082848121>, 1 }
    sphere {  m*<2.5024947524848886,-0.017337864269553388,-2.4667126850627343>, 1 }
    sphere {  m*<-1.8538290014142587,2.209102104762671,-2.211448925027521>, 1}
    sphere { m*<-1.5860417803764268,-2.6785898376412263,-2.0219026398649484>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.420820452607926,0.22977569908790096,6.866737082848121>, <-0.2322136415213687,-0.1193718396559276,-1.2375031596115538>, 0.5 }
    cylinder { m*<2.5024947524848886,-0.017337864269553388,-2.4667126850627343>, <-0.2322136415213687,-0.1193718396559276,-1.2375031596115538>, 0.5}
    cylinder { m*<-1.8538290014142587,2.209102104762671,-2.211448925027521>, <-0.2322136415213687,-0.1193718396559276,-1.2375031596115538>, 0.5 }
    cylinder {  m*<-1.5860417803764268,-2.6785898376412263,-2.0219026398649484>, <-0.2322136415213687,-0.1193718396559276,-1.2375031596115538>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    