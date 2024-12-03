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
    sphere { m*<-2.4917408310100378e-18,-8.995241546624977e-19,0.5299018393010515>, 1 }        
    sphere {  m*<-2.7952711619416325e-18,-5.327208150428884e-18,7.596901839301074>, 1 }
    sphere {  m*<9.428090415820634,-2.5262824034251796e-18,-2.8034314940322815>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.8034314940322815>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.8034314940322815>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.7952711619416325e-18,-5.327208150428884e-18,7.596901839301074>, <-2.4917408310100378e-18,-8.995241546624977e-19,0.5299018393010515>, 0.5 }
    cylinder { m*<9.428090415820634,-2.5262824034251796e-18,-2.8034314940322815>, <-2.4917408310100378e-18,-8.995241546624977e-19,0.5299018393010515>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.8034314940322815>, <-2.4917408310100378e-18,-8.995241546624977e-19,0.5299018393010515>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.8034314940322815>, <-2.4917408310100378e-18,-8.995241546624977e-19,0.5299018393010515>, 0.5}

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
    sphere { m*<-2.4917408310100378e-18,-8.995241546624977e-19,0.5299018393010515>, 1 }        
    sphere {  m*<-2.7952711619416325e-18,-5.327208150428884e-18,7.596901839301074>, 1 }
    sphere {  m*<9.428090415820634,-2.5262824034251796e-18,-2.8034314940322815>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.8034314940322815>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.8034314940322815>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.7952711619416325e-18,-5.327208150428884e-18,7.596901839301074>, <-2.4917408310100378e-18,-8.995241546624977e-19,0.5299018393010515>, 0.5 }
    cylinder { m*<9.428090415820634,-2.5262824034251796e-18,-2.8034314940322815>, <-2.4917408310100378e-18,-8.995241546624977e-19,0.5299018393010515>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.8034314940322815>, <-2.4917408310100378e-18,-8.995241546624977e-19,0.5299018393010515>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.8034314940322815>, <-2.4917408310100378e-18,-8.995241546624977e-19,0.5299018393010515>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    