## Frame-by-Frame Role Classification and Style Identification

##### **Grant Rhines \| July 2024**

Before anything else is said, I must give full credit to Devin Pleuler's [Fixed to Fluid: Frame-by-Frame Role Classification](https://github.com/devinpleuler/research/blob/master/frame-by-frame-position.md#fixed-to-fluid-frame-by-frame-role-classification) for the inspiration behind the format and methodology of the following research. The moment I saw the results of his work I wanted to recreate it and explore what I could add to the discussion.

------------------------------------------------------------------------

### Frame-by-Frame Role Classification

The fluid, unpredictable nature of soccer is a blessing for all of those who enjoy the sport and a curse for all of us who try to analyze it.

Assigning positional roles (center forward, right back, etc.) to players is one of the most fundamental steps in the analysis of soccer using data. By grouping players into these archetypes we are able to quickly attach context and expectations to performances at a match and seasonal level. A goal drought from a center midfielder is much more acceptable than a striker.

However, when trying to use traditional position labels to describe modern soccer tactics, the problems caused by their lack of flexibility is obvious.

For example, imagine that a player assigned as a left back is asked to play as a left winger when his team is in possession to allow the true left winger to come inside. On the other side of the field, the team's right back is told to stay in position and provide defensive cover. Does it really make sense to group these two players together as 'fullbacks'? Not really. (This issue of incongruity is even more pronounced when trying to compare outliers like Trent Alexander-Arnold to anyone else of his supposed position.)

Let's continue to demonstrate the issue at hand with a quick game. Two frames of tracking data are shown below that have been stripped of their assigned positions. How many positions can you guess correctly without any other context?

![](https://github.com/gkrhines/research/blob/main/src/guess_positions.png)

The frame on the left seems like a standard out-of-possession 4-4-2 formation and the frame on the right is the same team playing a 3-5-2 in possession. Now, let's see who is who.

![](https://github.com/gkrhines/research/blob/main/src/answers_positions.png)

In the example on the left, you probably didn't guess that right defensive midfielder has actually pulled wide and that the right winger is on the left side of field. In the right frame, you might not have gotten that the right winger is the one leading the line while the center forward has dropped deeper.

In both of the examples above, it would be ideal to shift our expectations based on how each player is actually behaving. In the first, we'd like to absolve the right defensive midfielder from some of the guilt if a valuable chance is conceded from midfield. In the second, we should expect the right winger to take and finish more shots when playing centrally.

While video and performance analysts are experts at detecting these trends when analyzing their team and the opposition, they are difficult to systematically identify on the scale that a modern recruitment department requires. A way to automatically detect the the position a player is playing on a match and season level would be beneficial for many reasons.

But this is not a new problem with soccer data and a lot of amazing research has been done on the subject. Like Plueler acknowledges in his implementation of frame-by-frame role classification, this project also borrows Shaw and Glickman's approach to summarizing a team's formation as a set of player-wise bivariate distributions like they used in Dynamic Analysis of Team Strategy in Professional Football[^frame-by-frame-position-cluster-1].

[^frame-by-frame-position-cluster-1]: <https://static.capabiliaserver.com/frontend/clients/barcanew/wp_prod/wp-content/uploads/2020/01/56ce723e-barca-conference-paper-laurie-shaw.pdf>

Gregory's Ready Player Run: Off-ball run identification and classification[^frame-by-frame-position-cluster-2] offers a simpler approach than Shaw and Glickman's that involves adjusting player positions relative to a team centroid. Again like Pleuler, I too followed the path of least resistance. As you can see from the visualizations below, this small tweak to the data set significantly decreases the covariance of each position's distribution.

[^frame-by-frame-position-cluster-2]: <https://static.capabiliaserver.com/frontend/clients/barca/wp_prod/wp-content/uploads/2020/01/ed15d067-ready-player-run-barcelona-paper-sam-gregory.pdf>

![](https://github.com/gkrhines/research/blob/main/src/centroid_adj_positions.png =1774x)

Though this is helpful to identify trends on the match level, its inability to help us add context to any specific moment in a match highlights what a frame-by-frame application would add to our analysis. So, let's create one!

To start, I will ask you to imagine that you are the right back in the following formation:

![](https://github.com/devinpleuler/research/blob/master/src/rb.png)

Besides the goalkeeper and central defenders, most of your teammates will usually be found higher up the field than you. You will often find yourself in line with the other wide defender. Except for the occasional winger, all of your teammates will be to your left when you are facing upfield.

As it turns out, all of the information you need to determine your current tactical responsibility can be inferred from your relative spatial positioning from your teammates.

To translate this spatial awareness into features that a computer can understand, let's imagine our player rotating in place and counting the number of teammates they can see in front of them each time he turns a little further.

![](https://github.com/devinpleuler/research/blob/master/src/rotate.gif)

When a goalkeeper is facing away from their goal, they will be able to see all ten of their teammates but will almost always not see anyone when looking toward their own goal.

On the other hand, a central midfielder will probably have a few teammates in sight no matter which direction they're facing but will rarely see all of them at once.

This relationship between the direction a player is facing and the number of visible teammates is represented in the following tables.

![](https://github.com/gkrhines/research/blob/main/src/teammate_counts.png =457x)

When compared side-by-side, it makes complete sense that players who play on the opposite sides of the field also have inverse relationships to one another. Furthermore, if we were to split each position's performance based on the half it was played in (and didn't normalize the direction of play), we would see a similar converse relationship.

With each row of the tables above as features, this information can be used to train a classification model that will assign a current position to each player on a frame-by-frame basis.

For a supervised classification approach, we also need labels to tell us what position each player was actually playing in each frame of tracking data. Luckily, most players occupy the roles that data providers assign to them for a majority of the game. Positional rotations are common, but they are mostly short-lived deviations from the team's normal structure.

Specifically, I used a XGBoost classifier model with default parameters to map the player-level spatial features to their positional labels.

Here are the frames from earlier and the model's predictions for each:

![](https://github.com/gkrhines/research/blob/main/src/true_positions-01.png)

![](https://github.com/gkrhines/research/blob/main/src/predictions.png)

The model incorrectly assigns a few labels, but they are all 'mistakes' that provide us with valuable information. Hopefully these predictions resemble the predictions you made at the beginning of this research!

**Left Frame**

-   The center and right defensive midfielders have pulled wide in midfield, and the model thinks they are right defensive midfielder and right winger, respectively.
-   The right winger has drifted to the left side of the field and has remained high upfield. Reasonably, the model has guessed they are a center forward.
-   The left winger is sitting a little deeper and playing very close to the team's center, almost inside the true left back. Therefore, the model has labelled them as a left back. (It would be interesting to see the results if the computer knew each label should be assigned only once and it chose between two players based on some confidence score.)

**Right Frame**

-   The right winger has moved into the highest and most central position, a clear indication of playing as a center forward in the model's opinion.

-   The center forward has dropped behind the right winger while staying in the middle of the field, and the model has labelled them as a center attacking midfielder.

We can construct a confusion matrix to judge how well the model performs when asked to make predictions about an entire game. Since most games of tracking data (that record data at 25 frames per second) feature over 3 million data points, this is no small feat.

![](https://github.com/gkrhines/research/blob/main/src/confusion_matrixes.jpg)

This is an incredibly helpful way to validate the model's performance. Team 1's right winger was labelled as a center forward more often than he was found to be behaving like a right winger. On the other side of the pitch, Team 1's left winger rarely found themselves leading the line. These observations should be relatively straightforward to validate or disprove by simply watching the game this tracking data is derived from.

As Pleuler points out, the fact that the model does not perfectly predict goalkeeper labels is evidence that we are not over-fitting the model. Since the inputs are purely geometric, it is easy to imagine the model getting confused by those rare situations when the goalkeeper isn't the closest player to their goal.

------------------------------------------------------------------------

### Style Identification

Though frame-by-frame position assignments add a lot of useful context to the raw tracking data, it is still far from being something that that can be presented to technical staff in either a recruitment or match analysis context. What we really need is to convert the assignments we have computed into a label that can be applied to players at the match or seasonal level.

Unfortunately, research focused on of identifying player style is sparse compared to the abundant amount of work that has been done to identify team style[^frame-by-frame-position-cluster-3]. The relatively low frequency of distinctive individual events in soccer also makes it difficult to interpret attempted actions as style, which is a common approach in other sports[^frame-by-frame-position-cluster-4]. Just because a winger is attempting a below average number of crosses doesn't necessarily mean he is playing as a inverted winger.

[^frame-by-frame-position-cluster-3]: <https://www.mdpi.com/2411-5142/8/3/104>

[^frame-by-frame-position-cluster-4]: <https://global-uploads.webflow.com/5f1af76ed86d6771ad48324b/5f6a65517f9440891b8e35d0_Kalman_NBA_Line_up_Analysis.pdf>

Given the shortcomings of using event data then, I think the positional assignments we have created are much more suited to attempt to capture player style statistically.

To do this, let's go back to the game we previously analyzed and focus on the left wingers who played for Team 1. Here are the centroid-adjusted average positions for each of them:

![](https://github.com/gkrhines/research/blob/main/src/lw_examples.png)

If we were just using the above information to identify player style, we would probably guess that it is more likely that Player 1 and Player 3 belong to the same group than Player 2 being grouped together with either of the others. This is helpful, but we are still a long way from being able to make any statements about the finer points of each performance.

Let's see if using each player's distribution of position assignments will help us determine which players should be grouped together and provide a more in depth analysis of each winger. Another confusion matrix should do the trick.

![](https://github.com/gkrhines/research/blob/main/src/lw_confusion_matrix.png =471x)

There are a few important takeaways from this.

-   Player 1 and Player 3's performances appear very similar statistically. This matches our expectation from their average positions.

-   We are able to learn valuable insights about each performance. For example, Player 3 was more likely play as a center forward and less likely to drop into a left back role than Player 1.

-   Our model is able to differentiate between different styles. If all of the player's role assignment distributions looked similar, that would be problematic.

These results are encouraging but we need a larger sample size than three to make any real judgment about the validity of using frame-by-frame position assignments to assess player style. So, we collected data from 22 more games where at least two left wingers made an appearance during the match. This resulted in 79 total player performances for us to analyze.

Next, we used the common approach of k-means clustering to group similar performances together. The analysis resulted in three centers being found in the data set. I've summarized the characteristics of each cluster below which ideally line up closely to styles that left wingers are known to use.

![](https://github.com/gkrhines/research/blob/main/src/cluster_high_low-01.png =407x)

Players grouped into Cluster 1 spent a significant more time playing on the left wing in both offensive and defensive roles. Additionally, those players spent much less time in the center of the field and almost no time on the right side. This is a strong match with the role commonly referred to as a "vertical winger".

Conversely, left wingers assigned to Cluster 3 usually spent less than half of their performances actually playing as left wingers. Instead, they often chose to come inside as a center forward or midfielder. They sometimes even ventured onto the right side of the field. This group sounds a lot like a stereotypical "inverted winger" to me.

To get a more visual understanding of the groupings, we can use a Principal Component Analysis to reduce the dimensionality of the data set and graph the results.

![](https://github.com/gkrhines/research/blob/main/src/clustering_results.png)

More solid results! There appears to be two distinctive types of players who play under the label in our sample of left wingers. It is also interesting to note that there is less covariance with the vertical wingers when compared to inverted wingers. Given that inverted wingers generally have much more tactical freedom than vertical wingers, this is great to see in the graph.

The analysis has also accidentally demonstrated the ability to use positional assignments to detect errors in the initial data set. Despite being categorized as a left winger by the data provider, the model labelled the only player performance in Cluster 2 as a defensive midfielder 71% of the time.

We now have what we came for! By applying the above methodology to get frame-by-frame positional assignments, aggregating them at the match level for each player, and using k-means clustering to group the players we have captured a purely statistical approach to assess player style. Most importantly, player-level style labels have a wide variety of applications to match analysis and player recruitment.

There is much more work to be done on this subject and hopefully this can contribute toward that.

------------------------------------------------------------------------

Again I would like acknowledge Devin Pleuler for his fantastic work and permission to use some of his visualizations to help explain his approach to frame-by-frame role classification.
