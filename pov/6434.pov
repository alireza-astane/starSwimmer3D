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
    sphere { m*<-1.2917603206335335,-0.6589162146258934,-0.871687943772809>, 1 }        
    sphere {  m*<0.1568494430394305,0.005282081219060403,9.000593863339223>, 1 }
    sphere {  m*<7.512200881039404,-0.08363819477529655,-5.578899426706128>, 1 }
    sphere {  m*<-4.851748068461146,3.8735114760220317,-2.695476510001936>, 1}
    sphere { m*<-2.6006533763299373,-3.2799731976529705,-1.5160357677784133>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1568494430394305,0.005282081219060403,9.000593863339223>, <-1.2917603206335335,-0.6589162146258934,-0.871687943772809>, 0.5 }
    cylinder { m*<7.512200881039404,-0.08363819477529655,-5.578899426706128>, <-1.2917603206335335,-0.6589162146258934,-0.871687943772809>, 0.5}
    cylinder { m*<-4.851748068461146,3.8735114760220317,-2.695476510001936>, <-1.2917603206335335,-0.6589162146258934,-0.871687943772809>, 0.5 }
    cylinder {  m*<-2.6006533763299373,-3.2799731976529705,-1.5160357677784133>, <-1.2917603206335335,-0.6589162146258934,-0.871687943772809>, 0.5}

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
    sphere { m*<-1.2917603206335335,-0.6589162146258934,-0.871687943772809>, 1 }        
    sphere {  m*<0.1568494430394305,0.005282081219060403,9.000593863339223>, 1 }
    sphere {  m*<7.512200881039404,-0.08363819477529655,-5.578899426706128>, 1 }
    sphere {  m*<-4.851748068461146,3.8735114760220317,-2.695476510001936>, 1}
    sphere { m*<-2.6006533763299373,-3.2799731976529705,-1.5160357677784133>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1568494430394305,0.005282081219060403,9.000593863339223>, <-1.2917603206335335,-0.6589162146258934,-0.871687943772809>, 0.5 }
    cylinder { m*<7.512200881039404,-0.08363819477529655,-5.578899426706128>, <-1.2917603206335335,-0.6589162146258934,-0.871687943772809>, 0.5}
    cylinder { m*<-4.851748068461146,3.8735114760220317,-2.695476510001936>, <-1.2917603206335335,-0.6589162146258934,-0.871687943772809>, 0.5 }
    cylinder {  m*<-2.6006533763299373,-3.2799731976529705,-1.5160357677784133>, <-1.2917603206335335,-0.6589162146258934,-0.871687943772809>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    