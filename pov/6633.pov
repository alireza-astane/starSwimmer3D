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
    sphere { m*<-1.1295718119364369,-0.8777181420691417,-0.7885232314269008>, 1 }        
    sphere {  m*<0.30970210105284846,-0.12132992107361973,9.07848506828807>, 1 }
    sphere {  m*<7.6650535390528125,-0.21025019706797662,-5.501008221757269>, 1 }
    sphere {  m*<-5.58950563251938,4.619415302548415,-3.0722494032302055>, 1}
    sphere { m*<-2.4007303828458704,-3.5218921648243513,-1.4136919213338863>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.30970210105284846,-0.12132992107361973,9.07848506828807>, <-1.1295718119364369,-0.8777181420691417,-0.7885232314269008>, 0.5 }
    cylinder { m*<7.6650535390528125,-0.21025019706797662,-5.501008221757269>, <-1.1295718119364369,-0.8777181420691417,-0.7885232314269008>, 0.5}
    cylinder { m*<-5.58950563251938,4.619415302548415,-3.0722494032302055>, <-1.1295718119364369,-0.8777181420691417,-0.7885232314269008>, 0.5 }
    cylinder {  m*<-2.4007303828458704,-3.5218921648243513,-1.4136919213338863>, <-1.1295718119364369,-0.8777181420691417,-0.7885232314269008>, 0.5}

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
    sphere { m*<-1.1295718119364369,-0.8777181420691417,-0.7885232314269008>, 1 }        
    sphere {  m*<0.30970210105284846,-0.12132992107361973,9.07848506828807>, 1 }
    sphere {  m*<7.6650535390528125,-0.21025019706797662,-5.501008221757269>, 1 }
    sphere {  m*<-5.58950563251938,4.619415302548415,-3.0722494032302055>, 1}
    sphere { m*<-2.4007303828458704,-3.5218921648243513,-1.4136919213338863>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.30970210105284846,-0.12132992107361973,9.07848506828807>, <-1.1295718119364369,-0.8777181420691417,-0.7885232314269008>, 0.5 }
    cylinder { m*<7.6650535390528125,-0.21025019706797662,-5.501008221757269>, <-1.1295718119364369,-0.8777181420691417,-0.7885232314269008>, 0.5}
    cylinder { m*<-5.58950563251938,4.619415302548415,-3.0722494032302055>, <-1.1295718119364369,-0.8777181420691417,-0.7885232314269008>, 0.5 }
    cylinder {  m*<-2.4007303828458704,-3.5218921648243513,-1.4136919213338863>, <-1.1295718119364369,-0.8777181420691417,-0.7885232314269008>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    