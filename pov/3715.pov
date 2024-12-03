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
    sphere { m*<0.007915924989628897,0.22284232379858793,-0.1234640201664402>, 1 }        
    sphere {  m*<0.24865102973132064,0.35155240197891335,2.864090750954112>, 1 }
    sphere {  m*<2.742624318995891,0.3248762991849625,-1.3526735456176269>, 1 }
    sphere {  m*<-1.6136994349032627,2.5513162682171897,-1.0974097855824123>, 1}
    sphere { m*<-2.243048993880038,-4.032284134184954,-1.4276588893344142>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.24865102973132064,0.35155240197891335,2.864090750954112>, <0.007915924989628897,0.22284232379858793,-0.1234640201664402>, 0.5 }
    cylinder { m*<2.742624318995891,0.3248762991849625,-1.3526735456176269>, <0.007915924989628897,0.22284232379858793,-0.1234640201664402>, 0.5}
    cylinder { m*<-1.6136994349032627,2.5513162682171897,-1.0974097855824123>, <0.007915924989628897,0.22284232379858793,-0.1234640201664402>, 0.5 }
    cylinder {  m*<-2.243048993880038,-4.032284134184954,-1.4276588893344142>, <0.007915924989628897,0.22284232379858793,-0.1234640201664402>, 0.5}

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
    sphere { m*<0.007915924989628897,0.22284232379858793,-0.1234640201664402>, 1 }        
    sphere {  m*<0.24865102973132064,0.35155240197891335,2.864090750954112>, 1 }
    sphere {  m*<2.742624318995891,0.3248762991849625,-1.3526735456176269>, 1 }
    sphere {  m*<-1.6136994349032627,2.5513162682171897,-1.0974097855824123>, 1}
    sphere { m*<-2.243048993880038,-4.032284134184954,-1.4276588893344142>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.24865102973132064,0.35155240197891335,2.864090750954112>, <0.007915924989628897,0.22284232379858793,-0.1234640201664402>, 0.5 }
    cylinder { m*<2.742624318995891,0.3248762991849625,-1.3526735456176269>, <0.007915924989628897,0.22284232379858793,-0.1234640201664402>, 0.5}
    cylinder { m*<-1.6136994349032627,2.5513162682171897,-1.0974097855824123>, <0.007915924989628897,0.22284232379858793,-0.1234640201664402>, 0.5 }
    cylinder {  m*<-2.243048993880038,-4.032284134184954,-1.4276588893344142>, <0.007915924989628897,0.22284232379858793,-0.1234640201664402>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    