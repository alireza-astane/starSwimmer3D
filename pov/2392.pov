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
    sphere { m*<0.9740506972974994,0.5137300656038806,0.44179090465802623>, 1 }        
    sphere {  m*<1.2178575853449771,0.5558442591076296,3.4315691730200957>, 1 }
    sphere {  m*<3.711104774407515,0.5558442591076294,-0.7857130354705231>, 1 }
    sphere {  m*<-2.7514625719655723,6.3289149032362655,-1.7609700650378721>, 1}
    sphere { m*<-3.822944380581021,-7.789199360275005,-2.3938293331085427>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2178575853449771,0.5558442591076296,3.4315691730200957>, <0.9740506972974994,0.5137300656038806,0.44179090465802623>, 0.5 }
    cylinder { m*<3.711104774407515,0.5558442591076294,-0.7857130354705231>, <0.9740506972974994,0.5137300656038806,0.44179090465802623>, 0.5}
    cylinder { m*<-2.7514625719655723,6.3289149032362655,-1.7609700650378721>, <0.9740506972974994,0.5137300656038806,0.44179090465802623>, 0.5 }
    cylinder {  m*<-3.822944380581021,-7.789199360275005,-2.3938293331085427>, <0.9740506972974994,0.5137300656038806,0.44179090465802623>, 0.5}

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
    sphere { m*<0.9740506972974994,0.5137300656038806,0.44179090465802623>, 1 }        
    sphere {  m*<1.2178575853449771,0.5558442591076296,3.4315691730200957>, 1 }
    sphere {  m*<3.711104774407515,0.5558442591076294,-0.7857130354705231>, 1 }
    sphere {  m*<-2.7514625719655723,6.3289149032362655,-1.7609700650378721>, 1}
    sphere { m*<-3.822944380581021,-7.789199360275005,-2.3938293331085427>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2178575853449771,0.5558442591076296,3.4315691730200957>, <0.9740506972974994,0.5137300656038806,0.44179090465802623>, 0.5 }
    cylinder { m*<3.711104774407515,0.5558442591076294,-0.7857130354705231>, <0.9740506972974994,0.5137300656038806,0.44179090465802623>, 0.5}
    cylinder { m*<-2.7514625719655723,6.3289149032362655,-1.7609700650378721>, <0.9740506972974994,0.5137300656038806,0.44179090465802623>, 0.5 }
    cylinder {  m*<-3.822944380581021,-7.789199360275005,-2.3938293331085427>, <0.9740506972974994,0.5137300656038806,0.44179090465802623>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    