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
    sphere { m*<0.5521958349242857,1.1237127776485416,0.19236546894821693>, 1 }        
    sphere {  m*<0.7934817984728075,1.2328380587187835,3.1806505333124306>, 1 }
    sphere {  m*<3.286728987535342,1.232838058718783,-1.0366316751781843>, 1 }
    sphere {  m*<-1.2881464900355728,3.775143743292137,-0.8957528751884922>, 1}
    sphere { m*<-3.9644895161799236,-7.388790749159042,-2.47752744628644>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7934817984728075,1.2328380587187835,3.1806505333124306>, <0.5521958349242857,1.1237127776485416,0.19236546894821693>, 0.5 }
    cylinder { m*<3.286728987535342,1.232838058718783,-1.0366316751781843>, <0.5521958349242857,1.1237127776485416,0.19236546894821693>, 0.5}
    cylinder { m*<-1.2881464900355728,3.775143743292137,-0.8957528751884922>, <0.5521958349242857,1.1237127776485416,0.19236546894821693>, 0.5 }
    cylinder {  m*<-3.9644895161799236,-7.388790749159042,-2.47752744628644>, <0.5521958349242857,1.1237127776485416,0.19236546894821693>, 0.5}

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
    sphere { m*<0.5521958349242857,1.1237127776485416,0.19236546894821693>, 1 }        
    sphere {  m*<0.7934817984728075,1.2328380587187835,3.1806505333124306>, 1 }
    sphere {  m*<3.286728987535342,1.232838058718783,-1.0366316751781843>, 1 }
    sphere {  m*<-1.2881464900355728,3.775143743292137,-0.8957528751884922>, 1}
    sphere { m*<-3.9644895161799236,-7.388790749159042,-2.47752744628644>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7934817984728075,1.2328380587187835,3.1806505333124306>, <0.5521958349242857,1.1237127776485416,0.19236546894821693>, 0.5 }
    cylinder { m*<3.286728987535342,1.232838058718783,-1.0366316751781843>, <0.5521958349242857,1.1237127776485416,0.19236546894821693>, 0.5}
    cylinder { m*<-1.2881464900355728,3.775143743292137,-0.8957528751884922>, <0.5521958349242857,1.1237127776485416,0.19236546894821693>, 0.5 }
    cylinder {  m*<-3.9644895161799236,-7.388790749159042,-2.47752744628644>, <0.5521958349242857,1.1237127776485416,0.19236546894821693>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    