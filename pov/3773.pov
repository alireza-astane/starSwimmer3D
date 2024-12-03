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
    sphere { m*<-0.02734227019031299,0.15619176078303676,-0.1438923959551191>, 1 }        
    sphere {  m*<0.2133928345513787,0.2849018389633622,2.8436623751654317>, 1 }
    sphere {  m*<2.7073661238159494,0.25822573616941136,-1.373101921406306>, 1 }
    sphere {  m*<-1.6489576300832045,2.4846657052016394,-1.1178381613710915>, 1}
    sphere { m*<-2.095089114268262,-3.7525871948064964,-1.3419318687954664>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2133928345513787,0.2849018389633622,2.8436623751654317>, <-0.02734227019031299,0.15619176078303676,-0.1438923959551191>, 0.5 }
    cylinder { m*<2.7073661238159494,0.25822573616941136,-1.373101921406306>, <-0.02734227019031299,0.15619176078303676,-0.1438923959551191>, 0.5}
    cylinder { m*<-1.6489576300832045,2.4846657052016394,-1.1178381613710915>, <-0.02734227019031299,0.15619176078303676,-0.1438923959551191>, 0.5 }
    cylinder {  m*<-2.095089114268262,-3.7525871948064964,-1.3419318687954664>, <-0.02734227019031299,0.15619176078303676,-0.1438923959551191>, 0.5}

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
    sphere { m*<-0.02734227019031299,0.15619176078303676,-0.1438923959551191>, 1 }        
    sphere {  m*<0.2133928345513787,0.2849018389633622,2.8436623751654317>, 1 }
    sphere {  m*<2.7073661238159494,0.25822573616941136,-1.373101921406306>, 1 }
    sphere {  m*<-1.6489576300832045,2.4846657052016394,-1.1178381613710915>, 1}
    sphere { m*<-2.095089114268262,-3.7525871948064964,-1.3419318687954664>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2133928345513787,0.2849018389633622,2.8436623751654317>, <-0.02734227019031299,0.15619176078303676,-0.1438923959551191>, 0.5 }
    cylinder { m*<2.7073661238159494,0.25822573616941136,-1.373101921406306>, <-0.02734227019031299,0.15619176078303676,-0.1438923959551191>, 0.5}
    cylinder { m*<-1.6489576300832045,2.4846657052016394,-1.1178381613710915>, <-0.02734227019031299,0.15619176078303676,-0.1438923959551191>, 0.5 }
    cylinder {  m*<-2.095089114268262,-3.7525871948064964,-1.3419318687954664>, <-0.02734227019031299,0.15619176078303676,-0.1438923959551191>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    