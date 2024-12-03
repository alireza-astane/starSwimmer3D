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
    sphere { m*<0.6727395642378875,0.9619045858883429,0.2636374906380613>, 1 }        
    sphere {  m*<0.9150041709774348,1.0512134196848255,3.2525019662011374>, 1 }
    sphere {  m*<3.4082513600399693,1.051213419684825,-0.9647802422894791>, 1 }
    sphere {  m*<-1.748964726105485,4.5340756722234845,-1.168219675937902>, 1}
    sphere { m*<-3.9276158388795053,-7.489900833094267,-2.4557235318501256>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9150041709774348,1.0512134196848255,3.2525019662011374>, <0.6727395642378875,0.9619045858883429,0.2636374906380613>, 0.5 }
    cylinder { m*<3.4082513600399693,1.051213419684825,-0.9647802422894791>, <0.6727395642378875,0.9619045858883429,0.2636374906380613>, 0.5}
    cylinder { m*<-1.748964726105485,4.5340756722234845,-1.168219675937902>, <0.6727395642378875,0.9619045858883429,0.2636374906380613>, 0.5 }
    cylinder {  m*<-3.9276158388795053,-7.489900833094267,-2.4557235318501256>, <0.6727395642378875,0.9619045858883429,0.2636374906380613>, 0.5}

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
    sphere { m*<0.6727395642378875,0.9619045858883429,0.2636374906380613>, 1 }        
    sphere {  m*<0.9150041709774348,1.0512134196848255,3.2525019662011374>, 1 }
    sphere {  m*<3.4082513600399693,1.051213419684825,-0.9647802422894791>, 1 }
    sphere {  m*<-1.748964726105485,4.5340756722234845,-1.168219675937902>, 1}
    sphere { m*<-3.9276158388795053,-7.489900833094267,-2.4557235318501256>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9150041709774348,1.0512134196848255,3.2525019662011374>, <0.6727395642378875,0.9619045858883429,0.2636374906380613>, 0.5 }
    cylinder { m*<3.4082513600399693,1.051213419684825,-0.9647802422894791>, <0.6727395642378875,0.9619045858883429,0.2636374906380613>, 0.5}
    cylinder { m*<-1.748964726105485,4.5340756722234845,-1.168219675937902>, <0.6727395642378875,0.9619045858883429,0.2636374906380613>, 0.5 }
    cylinder {  m*<-3.9276158388795053,-7.489900833094267,-2.4557235318501256>, <0.6727395642378875,0.9619045858883429,0.2636374906380613>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    