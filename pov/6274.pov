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
    sphere { m*<-1.4122105743673024,-0.48221134398330906,-0.9335375470908404>, 1 }        
    sphere {  m*<0.043602072779241,0.09969743697114158,8.942881138232227>, 1 }
    sphere {  m*<7.398953510779216,0.010777160976784661,-5.636612151813125>, 1 }
    sphere {  m*<-4.265032857505497,3.2566989014069483,-2.3956966341510566>, 1}
    sphere { m*<-2.754445367097409,-3.082096743915438,-1.5948363799712304>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.043602072779241,0.09969743697114158,8.942881138232227>, <-1.4122105743673024,-0.48221134398330906,-0.9335375470908404>, 0.5 }
    cylinder { m*<7.398953510779216,0.010777160976784661,-5.636612151813125>, <-1.4122105743673024,-0.48221134398330906,-0.9335375470908404>, 0.5}
    cylinder { m*<-4.265032857505497,3.2566989014069483,-2.3956966341510566>, <-1.4122105743673024,-0.48221134398330906,-0.9335375470908404>, 0.5 }
    cylinder {  m*<-2.754445367097409,-3.082096743915438,-1.5948363799712304>, <-1.4122105743673024,-0.48221134398330906,-0.9335375470908404>, 0.5}

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
    sphere { m*<-1.4122105743673024,-0.48221134398330906,-0.9335375470908404>, 1 }        
    sphere {  m*<0.043602072779241,0.09969743697114158,8.942881138232227>, 1 }
    sphere {  m*<7.398953510779216,0.010777160976784661,-5.636612151813125>, 1 }
    sphere {  m*<-4.265032857505497,3.2566989014069483,-2.3956966341510566>, 1}
    sphere { m*<-2.754445367097409,-3.082096743915438,-1.5948363799712304>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.043602072779241,0.09969743697114158,8.942881138232227>, <-1.4122105743673024,-0.48221134398330906,-0.9335375470908404>, 0.5 }
    cylinder { m*<7.398953510779216,0.010777160976784661,-5.636612151813125>, <-1.4122105743673024,-0.48221134398330906,-0.9335375470908404>, 0.5}
    cylinder { m*<-4.265032857505497,3.2566989014069483,-2.3956966341510566>, <-1.4122105743673024,-0.48221134398330906,-0.9335375470908404>, 0.5 }
    cylinder {  m*<-2.754445367097409,-3.082096743915438,-1.5948363799712304>, <-1.4122105743673024,-0.48221134398330906,-0.9335375470908404>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    